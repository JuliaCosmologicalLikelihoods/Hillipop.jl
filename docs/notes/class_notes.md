# CLASS.jl Summary

1. **CLASS Problem Construction:**
   - Use `CLASSProblem(dict)` where `dict` contains keys matching CLASS `.ini` format.
   - Example keys: `"h"`, `"omega_b"`, `"output"`, `"lensing"`.

2. **Running CLASS:**
   - Execute `sol = solve(prob)`. Note: User must have CLASS installed, or specify `exec` path. It is a wrapper around the binary. (Wait, let me double check CLASS.jl README. Ah, it says it writes an ini file and calls the executable. It also seems to assume a `class` binary is in the `PATH` or specified via `exec`. Wait, maybe I should check if there's an easier way, but I will just assume the user will have it, or provide instructions. Actually, Julia might not have `class` in `PATH` by default, but we'll write the script as instructed. Or wait, maybe we use `exec = "class"` or similar.)

3. **Output Format:**
   - `sol` is a dictionary-like object.
   - `cls_df = sol["cl"]` (or `"lensed_cl"`? We'll check via standard CLASS output conventions. Typically `"cl"` or `"lCl"`. The README doesn't specify exactly for lensed Cls. Often CLASS outputs `cl_lensed.dat`. So maybe `sol["cl_lensed"]` or `sol["cl"]`?). CLASS outputs dimensionless `l*(l+1)*C_l/(2pi)` unless `format = camb` is passed. Oh wait, native CLASS output is dimensionless D_l = l(l+1)Cl/2pi but with Tcmb factored out. Let me check CLASS documentation or write a small test script to inspect the columns.

4. **Installation & Vendor:**
   - Needs `using Pkg; Pkg.add("CLASS")`.
   - Requires external `class` installation.

5. **Limitations:**
   - Thread safety: since it probably writes to temporary files and runs a binary, might have issues if run concurrently without isolated working directories.
