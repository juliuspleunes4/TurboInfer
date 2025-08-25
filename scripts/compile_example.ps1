# TurboInfer Example Compiler
# Usage: .\scripts\compile_example.ps1 [example_name]
# Example: .\scripts\compile_example.ps1 readme_example

param(
    [string]$ExampleName = ""
)

function Write-Banner($Message) {
    Write-Host "=" * 50 -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "=" * 50 -ForegroundColor Cyan
}

function Show-Help {
    Write-Banner "TurboInfer Example Compiler"
    Write-Host "Usage: .\scripts\compile_example.ps1 [example_name]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Available examples:" -ForegroundColor White
    Get-ChildItem examples/*.cpp | ForEach-Object {
        $name = [System.IO.Path]::GetFileNameWithoutExtension($_.Name)
        Write-Host "  $name" -ForegroundColor Green
    }
    Write-Host ""
    Write-Host "Example usage:" -ForegroundColor White
    Write-Host "  .\scripts\compile_example.ps1 readme_example" -ForegroundColor Gray
}

function Build-Example($name) {
    Write-Banner "Compiling Example: $name"
    
    $sourceFile = "examples/$name.cpp"
    $outputFile = "examples/$name.exe"
    $libraryFile = "build/lib/libturboinfer.a"
    
    # Check if source file exists
    if (-not (Test-Path $sourceFile)) {
        Write-Host "❌ Source file not found: $sourceFile" -ForegroundColor Red
        return
    }
    
    # Check if library exists
    if (-not (Test-Path $libraryFile)) {
        Write-Host "❌ Library not found: $libraryFile" -ForegroundColor Red
        Write-Host "Please build the project first with: .\scripts\dev.ps1 build" -ForegroundColor Yellow
        return
    }
    
    # Compile
    $compileCmd = "g++ -std=c++20 -I include $sourceFile $libraryFile -o $outputFile"
    Write-Host "Compiling: $compileCmd" -ForegroundColor Yellow
    
    Invoke-Expression $compileCmd
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Compilation successful!" -ForegroundColor Green
        Write-Host "Run with: .\$outputFile" -ForegroundColor Gray
    } else {
        Write-Host "❌ Compilation failed!" -ForegroundColor Red
    }
}

# Main execution
if ($ExampleName -eq "") {
    Show-Help
} else {
    Build-Example $ExampleName
}
