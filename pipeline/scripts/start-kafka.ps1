<#
.SYNOPSIS
    Run a single-node Kafka broker natively on Windows, for machines without a working Docker.

.DESCRIPTION
    Downloads Apache Kafka (once, checksum-verified) into pipeline\.kafka, formats its storage
    (once), and runs the broker in the foreground on localhost:9092. Press Ctrl+C to stop it.
    Needs Java 17 or newer on PATH. Equivalent to `docker compose up` with docker-compose.yml.
#>
param(
    [string]$Version = "4.3.1",
    [switch]$Wipe       # delete all broker data (topics, messages, offsets) before starting
)
$ErrorActionPreference = "Stop"

$root = Join-Path $PSScriptRoot "..\.kafka" | Resolve-Path -ErrorAction SilentlyContinue
if (-not $root) { $root = New-Item -ItemType Directory -Path (Join-Path $PSScriptRoot "..\.kafka") }
$root = "$root"
$name = "kafka_2.13-$Version"
$kafkaHome = Join-Path $root $name
$dataDir = Join-Path $root "data"
$config = Join-Path $root "server.properties"

if (-not (Test-Path $kafkaHome)) {
    $url = "https://downloads.apache.org/kafka/$Version/$name.tgz"
    $tgz = Join-Path $root "$name.tgz"
    Write-Host "Downloading $url"
    Invoke-WebRequest $url -OutFile $tgz
    $expected = ((Invoke-WebRequest "$url.sha512").Content -replace "\s", "" -split ":")[-1].ToLower()
    $actual = (Get-FileHash $tgz -Algorithm SHA512).Hash.ToLower()
    if ($expected -ne $actual) { throw "checksum mismatch for $tgz" }
    tar -xzf $tgz -C $root
    Remove-Item $tgz
}

if ($Wipe -and (Test-Path $dataDir)) { Remove-Item -Recurse -Force $dataDir }

# KRaft single node: this process is both broker and controller. Same settings as docker-compose.yml.
@"
process.roles=broker,controller
node.id=1
controller.quorum.bootstrap.servers=localhost:9093
listeners=PLAINTEXT://localhost:9092,CONTROLLER://localhost:9093
advertised.listeners=PLAINTEXT://localhost:9092
controller.listener.names=CONTROLLER
listener.security.protocol.map=CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT
log.dirs=$($dataDir -replace '\\', '/')
num.partitions=1
offsets.topic.replication.factor=1
transaction.state.log.replication.factor=1
transaction.state.log.min.isr=1
group.initial.rebalance.delay.ms=0
auto.create.topics.enable=false
"@ | Set-Content -Encoding ascii $config

# Call Java directly with a wildcard classpath: the bundled .bat launchers build a classpath that
# can exceed the Windows command-line limit.
$classpath = Join-Path $kafkaHome "libs\*"
$log4j = "-Dlog4j2.configurationFile=$(Join-Path $kafkaHome 'config\log4j2.yaml')"
$logsDir = "-Dkafka.logs.dir=$(Join-Path $root 'logs')"

if (-not (Test-Path (Join-Path $dataDir "meta.properties"))) {
    $clusterId = (& java -cp $classpath kafka.tools.StorageTool random-uuid).Trim()
    & java -cp $classpath $log4j $logsDir kafka.tools.StorageTool format --standalone -t $clusterId -c $config
    if ($LASTEXITCODE -ne 0) { throw "formatting Kafka storage failed" }
}

Write-Host "Starting Kafka $Version on localhost:9092 (Ctrl+C to stop)"
& java -Xmx1G -cp $classpath $log4j $logsDir kafka.Kafka $config
