import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot049Data
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot049Part00
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot049Part01
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
theorem root49Scan33 : scan edge (search edge 3) 4 [239] 239 root49Tail33 winner = winner := by rw [root49Tail33, scan, if_pos (by decide)]
theorem root49Scan32 : scan edge (search edge 3) 4 [239] 239 root49Tail32 winner = winner :=
  (root49Step32).trans root49Scan33
theorem root49Scan31 : scan edge (search edge 3) 4 [239] 239 root49Tail31 winner = winner :=
  (root49Step31).trans root49Scan32
theorem root49Scan30 : scan edge (search edge 3) 4 [239] 239 root49Tail30 winner = winner :=
  (root49Step30).trans root49Scan31
theorem root49Scan29 : scan edge (search edge 3) 4 [239] 239 root49Tail29 winner = winner :=
  (root49Step29).trans root49Scan30
theorem root49Scan28 : scan edge (search edge 3) 4 [239] 239 root49Tail28 winner = winner :=
  (root49Step28).trans root49Scan29
theorem root49Scan27 : scan edge (search edge 3) 4 [239] 239 root49Tail27 winner = winner :=
  (root49Step27).trans root49Scan28
theorem root49Scan26 : scan edge (search edge 3) 4 [239] 239 root49Tail26 winner = winner :=
  (root49Step26).trans root49Scan27
theorem root49Scan25 : scan edge (search edge 3) 4 [239] 239 root49Tail25 winner = winner :=
  (root49Step25).trans root49Scan26
theorem root49Scan24 : scan edge (search edge 3) 4 [239] 239 root49Tail24 winner = winner :=
  (root49Step24).trans root49Scan25
theorem root49Scan23 : scan edge (search edge 3) 4 [239] 239 root49Tail23 winner = winner :=
  (root49Step23).trans root49Scan24
theorem root49Scan22 : scan edge (search edge 3) 4 [239] 239 root49Tail22 winner = winner :=
  (root49Step22).trans root49Scan23
theorem root49Scan21 : scan edge (search edge 3) 4 [239] 239 root49Tail21 winner = winner :=
  (root49Step21).trans root49Scan22
theorem root49Scan20 : scan edge (search edge 3) 4 [239] 239 root49Tail20 winner = winner :=
  (root49Step20).trans root49Scan21
theorem root49Scan19 : scan edge (search edge 3) 4 [239] 239 root49Tail19 winner = winner :=
  (root49Step19).trans root49Scan20
theorem root49Scan18 : scan edge (search edge 3) 4 [239] 239 root49Tail18 winner = winner :=
  (root49Step18).trans root49Scan19
theorem root49Scan17 : scan edge (search edge 3) 4 [239] 239 root49Tail17 winner = winner :=
  (root49Step17).trans root49Scan18
theorem root49Scan16 : scan edge (search edge 3) 4 [239] 239 root49Tail16 winner = winner :=
  (root49Step16).trans root49Scan17
theorem root49Scan15 : scan edge (search edge 3) 4 [239] 239 root49Tail15 winner = winner :=
  (root49Step15).trans root49Scan16
theorem root49Scan14 : scan edge (search edge 3) 4 [239] 239 root49Tail14 winner = winner :=
  (root49Step14).trans root49Scan15
theorem root49Scan13 : scan edge (search edge 3) 4 [239] 239 root49Tail13 winner = winner :=
  (root49Step13).trans root49Scan14
theorem root49Scan12 : scan edge (search edge 3) 4 [239] 239 root49Tail12 winner = winner :=
  (root49Step12).trans root49Scan13
theorem root49Scan11 : scan edge (search edge 3) 4 [239] 239 root49Tail11 winner = winner :=
  (root49Step11).trans root49Scan12
theorem root49Scan10 : scan edge (search edge 3) 4 [239] 239 root49Tail10 winner = winner :=
  (root49Step10).trans root49Scan11
theorem root49Scan9 : scan edge (search edge 3) 4 [239] 239 root49Tail9 winner = winner :=
  (root49Step9).trans root49Scan10
theorem root49Scan8 : scan edge (search edge 3) 4 [239] 239 root49Tail8 winner = winner :=
  (root49Step8).trans root49Scan9
theorem root49Scan7 : scan edge (search edge 3) 4 [239] 239 root49Tail7 winner = winner :=
  (root49Step7).trans root49Scan8
theorem root49Scan6 : scan edge (search edge 3) 4 [239] 239 root49Tail6 winner = winner :=
  (root49Step6).trans root49Scan7
theorem root49Scan5 : scan edge (search edge 3) 4 [239] 239 root49Tail5 winner = winner :=
  (root49Step5).trans root49Scan6
theorem root49Scan4 : scan edge (search edge 3) 4 [239] 239 root49Tail4 winner = winner :=
  (root49Step4).trans root49Scan5
theorem root49Scan3 : scan edge (search edge 3) 4 [239] 239 root49Tail3 winner = winner :=
  (root49Step3).trans root49Scan4
theorem root49Scan2 : scan edge (search edge 3) 4 [239] 239 root49Tail2 winner = winner :=
  (root49Step2).trans root49Scan3
theorem root49Scan1 : scan edge (search edge 3) 4 [239] 239 root49Tail1 winner = winner :=
  (root49Step1).trans root49Scan2
theorem root49Scan0 : scan edge (search edge 3) 4 [239] 239 root49Tail0 winner = winner :=
  (root49Step0).trans root49Scan1
set_option maxRecDepth 4096 in
theorem root49Branch : search edge 4 [239] root49Candidates 239 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [239] 239 root49Tail0 winner = winner
  exact root49Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
