# QAIRT Virtual Manager
- The qairt-vm CLI is the QAIRT Dev environment management tool.
- It inspects your host, applies recommended fixes to our environment, and fetches compatible QAIRT SDK releases.
step-1
# Inspect (no changes)
  qairt-vm -i -v
# List available compatible SDK versions
  qairt-vm fetch -l
# Fetch a specific SDK version to a custom directory
qairt-vm fetch -v 2.39.0 -d /tmp/qairt_sdk
# Point QAIRT to custom SDK root (Linux example)
export QAIRT_SDK_ROOT=/tmp/qairt_sdk/qairt/2.39.0.<build_number>
# Apply fixes interactively (shows prompts)
qairt-vm -f -v
# Apply all fixes without prompts (CI/automation)
qairt-vm -f -y -v
# Re-run inspection
qairt-vm -i
# Programmatic verification
python -c "import qairt; print(qairt.__dev_version__, qairt.__sdk_version__)"
