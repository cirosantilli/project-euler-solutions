import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
theorem outerScan690 : scan edge (search edge 4) 5 [] 0 tail690 winner = winner := by
  rw [tail690, scan, if_pos (by decide)]
end ProjectEulerSolutions.PrimePairLarge.Search
