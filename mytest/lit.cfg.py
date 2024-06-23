# based on https://github.com/openxla/xla/blob/main/xla/lit.cfg.py

# See the License for the specific language governing permissions and
# limitations under the License.
"""Lit runner configuration."""

import os
import sys
import tempfile

import lit.formats


# pylint: disable=undefined-variable


config.name = "XLA"
config.suffixes = [".cc", ".hlo", ".json", ".mlir", ".pbtxt", ".py"]

config.test_format = lit.formats.ShTest(execute_external=True)


# Passthrough XLA_FLAGS.
config.environment["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "")

# Use the most preferred temp directory.
config.test_exec_root = (
    os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    or os.environ.get("TEST_TMPDIR")
    or os.path.join(tempfile.gettempdir(), "lit")
)

config.substitutions.extend([
    ("%PYTHON", os.getenv("PYTHON", sys.executable)),
])

# Include additional substitutions that may be defined via params
config.substitutions.extend(
    ("%%{%s}" % key, val)
    for key, val in lit_config.params.items()
)