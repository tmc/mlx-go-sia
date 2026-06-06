//go:build unix

package sia

import "syscall"

// noFollow makes os.OpenFile refuse to open a symlinked leaf, closing the
// "fixture is a symlink to a frozen oracle" gaming vector. safeJoin already
// rejects symlink components that escape the gen dir; this guards the final
// component too.
const noFollow = syscall.O_NOFOLLOW
