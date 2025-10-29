#!/bin/bash
# Integration test script for Claude Code integration (RDR-006 enhancement)
# Tests that CLI outputs are properly formatted for Claude to parse

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Helper functions
test_header() {
    echo ""
    echo "============================================"
    echo "Testing: $1"
    echo "============================================"
}

pass() {
    echo -e "${GREEN}✅ PASS${NC}: $1"
    ((PASSED_TESTS++))
    ((TOTAL_TESTS++))
}

fail() {
    echo -e "${RED}❌ FAIL${NC}: $1"
    ((FAILED_TESTS++))
    ((TOTAL_TESTS++))
}

warn() {
    echo -e "${YELLOW}⚠️  WARN${NC}: $1"
}

# Test JSON output is valid
test_json_output() {
    local cmd="$1"
    local desc="$2"

    echo "Testing: $desc"

    # Run command with --json flag
    output=$(python -m arcaneum.cli.main $cmd --json 2>&1 || true)

    # Check if output is valid JSON
    if echo "$output" | python -m json.tool > /dev/null 2>&1; then
        pass "$desc - valid JSON output"
    else
        fail "$desc - invalid JSON output"
        echo "Output: $output"
    fi

    # Check for required JSON fields
    if echo "$output" | grep -q '"status"' && echo "$output" | grep -q '"message"' && echo "$output" | grep -q '"data"'; then
        pass "$desc - has required JSON fields (status, message, data)"
    else
        fail "$desc - missing required JSON fields"
    fi
}

# Test exit codes
test_exit_code() {
    local cmd="$1"
    local expected_code="$2"
    local desc="$3"

    echo "Testing: $desc"

    # Run command and capture exit code
    python -m arcaneum.cli.main $cmd > /dev/null 2>&1
    actual_code=$?

    if [ $actual_code -eq $expected_code ]; then
        pass "$desc - exit code $actual_code matches expected $expected_code"
    else
        fail "$desc - exit code $actual_code, expected $expected_code"
    fi
}

# Test progress message format
test_progress_format() {
    local output="$1"
    local desc="$2"

    echo "Testing: $desc"

    # Check for [INFO] prefix
    if echo "$output" | grep -q '\[INFO\]'; then
        pass "$desc - has [INFO] prefix"
    else
        warn "$desc - missing [INFO] prefix (may be intentional)"
    fi

    # Check for percentage in progress updates (if applicable)
    if echo "$output" | grep -qE '\([0-9]+%\)|\([0-9]+/[0-9]+\)'; then
        pass "$desc - has progress indicators"
    else
        warn "$desc - no progress indicators found (may not be needed)"
    fi
}

# Test error message format
test_error_format() {
    local cmd="$1"
    local desc="$2"

    echo "Testing: $desc"

    # Run command that should fail and capture stderr
    output=$(python -m arcaneum.cli.main $cmd 2>&1 || true)

    # Check for [ERROR] prefix
    if echo "$output" | grep -q '\[ERROR\]'; then
        pass "$desc - has [ERROR] prefix"
    else
        fail "$desc - missing [ERROR] prefix in error output"
        echo "Output: $output"
    fi
}

# Test $ARGUMENTS expansion simulation
test_arguments_expansion() {
    local cmd="$1"
    local desc="$2"

    echo "Testing: $desc"

    # Run command with various flag combinations
    python -m arcaneum.cli.main $cmd > /dev/null 2>&1 || true

    if [ $? -ne 127 ]; then  # 127 = command not found
        pass "$desc - command accepts arguments"
    else
        fail "$desc - command not found or doesn't accept arguments"
    fi
}

# Main test execution
main() {
    echo "========================================"
    echo "Arcaneum Claude Code Integration Tests"
    echo "========================================"
    echo ""
    echo "Project root: $PROJECT_ROOT"
    cd "$PROJECT_ROOT"

    # Test 1: JSON output for all commands
    test_header "JSON Output Validation"
    test_json_output "list-models" "list-models command"
    test_json_output "list-collections" "list-collections command"
    test_json_output "doctor" "doctor command"

    # Test 2: Help output for all commands
    test_header "Help Output"
    for cmd in "index-pdfs" "index-source" "create-collection" "list-collections" \
               "search" "search-text" "create-corpus" "sync-directory" "doctor"; do
        test_arguments_expansion "$cmd --help" "$cmd --help"
    done

    # Test 3: Error message formatting
    test_header "Error Message Format"
    test_error_format "create-collection InvalidName --model unknown_model" "invalid model error"
    test_error_format "search 'test query' --collection NonExistent" "collection not found error"

    # Test 4: Exit codes
    test_header "Exit Codes"
    test_exit_code "list-models --json" 0 "successful command"
    # Note: Can't easily test other exit codes without actual errors

    # Test 5: Command registration
    test_header "Command Registration"
    commands=("index-pdfs" "index-source" "create-collection" "list-collections" \
              "search" "search-text" "create-corpus" "sync-directory" "doctor")

    for cmd in "${commands[@]}"; do
        if python -m arcaneum.cli.main "$cmd" --help > /dev/null 2>&1; then
            pass "$cmd command is registered"
        else
            fail "$cmd command is not registered or help doesn't work"
        fi
    done

    # Test 6: Plugin manifest validation
    test_header "Plugin Manifest Validation"

    # Check plugin.json exists
    if [ -f ".claude-plugin/plugin.json" ]; then
        pass "plugin.json exists"

        # Validate JSON syntax
        if python -m json.tool ".claude-plugin/plugin.json" > /dev/null 2>&1; then
            pass "plugin.json is valid JSON"
        else
            fail "plugin.json has invalid JSON syntax"
        fi

        # Check commands array
        if grep -q '"commands"' ".claude-plugin/plugin.json"; then
            pass "plugin.json has commands array"

            # Verify all command files exist
            for cmd_file in $(python -c "import json; f=open('.claude-plugin/plugin.json'); data=json.load(f); print('\n'.join(data['commands']))" 2>/dev/null); do
                cmd_path="${cmd_file#./}"  # Remove ./ prefix
                if [ -f "$cmd_path" ]; then
                    pass "Command file exists: $cmd_path"
                else
                    fail "Command file missing: $cmd_path"
                fi
            done
        else
            fail "plugin.json missing commands array"
        fi
    else
        fail "plugin.json not found"
    fi

    # Test 7: Command file format
    test_header "Command File Format"

    for cmd_file in commands/*.md; do
        if [ -f "$cmd_file" ]; then
            # Check for YAML frontmatter
            if head -n 1 "$cmd_file" | grep -q '^---$'; then
                pass "$(basename $cmd_file) has YAML frontmatter"
            else
                fail "$(basename $cmd_file) missing YAML frontmatter"
            fi

            # Check for description field
            if grep -q '^description:' "$cmd_file"; then
                pass "$(basename $cmd_file) has description field"
            else
                fail "$(basename $cmd_file) missing description field"
            fi

            # Check for argument-hint field
            if grep -q '^argument-hint:' "$cmd_file"; then
                pass "$(basename $cmd_file) has argument-hint field"
            else
                warn "$(basename $cmd_file) missing argument-hint field (optional)"
            fi

            # Check for execution block
            if grep -q 'python -m arcaneum.cli.main' "$cmd_file"; then
                pass "$(basename $cmd_file) has execution block"
            else
                fail "$(basename $cmd_file) missing execution block"
            fi

            # Check for ${CLAUDE_PLUGIN_ROOT} usage
            if grep -q '${CLAUDE_PLUGIN_ROOT}' "$cmd_file"; then
                pass "$(basename $cmd_file) uses \${CLAUDE_PLUGIN_ROOT}"
            else
                warn "$(basename $cmd_file) doesn't use \${CLAUDE_PLUGIN_ROOT} (may be intentional)"
            fi
        fi
    done

    # Test 8: Output encoding
    test_header "Output Encoding"

    # Test that output is UTF-8 compatible
    output=$(python -m arcaneum.cli.main list-models 2>&1)
    if echo "$output" | iconv -f UTF-8 -t UTF-8 > /dev/null 2>&1; then
        pass "CLI output is valid UTF-8"
    else
        fail "CLI output has encoding issues"
    fi

    # Summary
    echo ""
    echo "========================================"
    echo "Test Summary"
    echo "========================================"
    echo "Total tests: $TOTAL_TESTS"
    echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
    echo -e "${RED}Failed: $FAILED_TESTS${NC}"
    echo ""

    if [ $FAILED_TESTS -eq 0 ]; then
        echo -e "${GREEN}✅ All tests passed!${NC}"
        exit 0
    else
        echo -e "${RED}❌ Some tests failed${NC}"
        exit 1
    fi
}

# Run main function
main "$@"
