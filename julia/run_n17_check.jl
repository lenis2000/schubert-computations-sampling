#!/usr/bin/env julia
#
# Independent Julia cross-check at n=17 of the certified C++ values of
# the principal specialization Upsilon_w = S_w(1, ..., 1) for the three
# permutations relevant to the disproof of the Merzon-Smirnov conjecture
# in Anderson-Panova-Petrov, "Computation and sampling for Schubert
# specializations".
#
# Backend: SchubertPolynomials.jl (Anderson, OSU), function nschub.
# Reference: https://github.com/pseudoeffective/SchubertPolynomials.jl
#
# Usage:
#   julia --project=. run_n17_check.jl
#
# First-time setup (in this directory):
#   julia --project=. -e 'using Pkg; Pkg.add(url="https://github.com/pseudoeffective/SchubertPolynomials.jl")'

using SchubertPolynomials
using Pkg
using Dates

# Three n=17 permutations (one-line notation). The first two are the
# counterexamples to the Merzon-Smirnov conjecture; the third is the
# optimal layered permutation w(1,2,4,10) that w* and u* beat.
const PERMS = [
    ("w*       (counterexample)",
     [1,3,2,7,6,5,17,4,16,15,14,13,12,11,10,9,8],
     big"3272424600397137120000"),
    ("u*       (counterexample)",
     [1,3,2,8,6,5,17,4,16,15,14,13,12,11,10,9,7],
     big"3528445515842977489500"),
    ("w(1,2,4,10) (layered max)",
     [1,3,2,7,6,5,4,17,16,15,14,13,12,11,10,9,8],
     big"3050684475186219300000"),
]

println("# Julia cross-check at n=17")
println("# Julia version: ", VERSION)
let pkg_uuid = Base.UUID("ca6918c7-c46a-4663-8c75-d24c3a15490e")
    deps = Pkg.dependencies()
    if haskey(deps, pkg_uuid)
        info = deps[pkg_uuid]
        println("# SchubertPolynomials.jl: ", info.version)
    end
end
println("# Date: ", Dates.format(now(UTC), "yyyy-mm-dd HH:MM:SS"), " UTC")
println()

all_match = true
for (name, w, expected) in PERMS
    t = @elapsed v = nschub(w)
    match = (v == expected)
    global all_match = all_match && match
    status = match ? "OK" : "MISMATCH"
    println("  $name")
    println("    nschub   = $v")
    println("    expected = $expected")
    println("    $status   ($(round(t, digits=2)) s)")
    println()
end

if all_match
    println("ALL THREE n=17 VALUES MATCH C++ CERTIFICATION (digit-for-digit).")
else
    println("MISMATCH detected. Investigate.")
    exit(1)
end
