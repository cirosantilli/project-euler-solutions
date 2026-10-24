import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot022Data
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot022Part00
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot022Part01
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
theorem root22Scan37 : scan edge (search edge 3) 4 [97] 97 root22Tail37 winner = winner := by rw [root22Tail37, scan, if_pos (by decide)]
theorem root22Scan36 : scan edge (search edge 3) 4 [97] 97 root22Tail36 winner = winner :=
  (root22Step36).trans root22Scan37
theorem root22Scan35 : scan edge (search edge 3) 4 [97] 97 root22Tail35 winner = winner :=
  (root22Step35).trans root22Scan36
theorem root22Scan34 : scan edge (search edge 3) 4 [97] 97 root22Tail34 winner = winner :=
  (root22Step34).trans root22Scan35
theorem root22Scan33 : scan edge (search edge 3) 4 [97] 97 root22Tail33 winner = winner :=
  (root22Step33).trans root22Scan34
theorem root22Scan32 : scan edge (search edge 3) 4 [97] 97 root22Tail32 winner = winner :=
  (root22Step32).trans root22Scan33
theorem root22Scan31 : scan edge (search edge 3) 4 [97] 97 root22Tail31 winner = winner :=
  (root22Step31).trans root22Scan32
theorem root22Scan30 : scan edge (search edge 3) 4 [97] 97 root22Tail30 winner = winner :=
  (root22Step30).trans root22Scan31
theorem root22Scan29 : scan edge (search edge 3) 4 [97] 97 root22Tail29 winner = winner :=
  (root22Step29).trans root22Scan30
theorem root22Scan28 : scan edge (search edge 3) 4 [97] 97 root22Tail28 winner = winner :=
  (root22Step28).trans root22Scan29
theorem root22Scan27 : scan edge (search edge 3) 4 [97] 97 root22Tail27 winner = winner :=
  (root22Step27).trans root22Scan28
theorem root22Scan26 : scan edge (search edge 3) 4 [97] 97 root22Tail26 winner = winner :=
  (root22Step26).trans root22Scan27
theorem root22Scan25 : scan edge (search edge 3) 4 [97] 97 root22Tail25 winner = winner :=
  (root22Step25).trans root22Scan26
theorem root22Scan24 : scan edge (search edge 3) 4 [97] 97 root22Tail24 winner = winner :=
  (root22Step24).trans root22Scan25
theorem root22Scan23 : scan edge (search edge 3) 4 [97] 97 root22Tail23 winner = winner :=
  (root22Step23).trans root22Scan24
theorem root22Scan22 : scan edge (search edge 3) 4 [97] 97 root22Tail22 winner = winner :=
  (root22Step22).trans root22Scan23
theorem root22Scan21 : scan edge (search edge 3) 4 [97] 97 root22Tail21 winner = winner :=
  (root22Step21).trans root22Scan22
theorem root22Scan20 : scan edge (search edge 3) 4 [97] 97 root22Tail20 winner = winner :=
  (root22Step20).trans root22Scan21
theorem root22Scan19 : scan edge (search edge 3) 4 [97] 97 root22Tail19 winner = winner :=
  (root22Step19).trans root22Scan20
theorem root22Scan18 : scan edge (search edge 3) 4 [97] 97 root22Tail18 winner = winner :=
  (root22Step18).trans root22Scan19
theorem root22Scan17 : scan edge (search edge 3) 4 [97] 97 root22Tail17 winner = winner :=
  (root22Step17).trans root22Scan18
theorem root22Scan16 : scan edge (search edge 3) 4 [97] 97 root22Tail16 winner = winner :=
  (root22Step16).trans root22Scan17
theorem root22Scan15 : scan edge (search edge 3) 4 [97] 97 root22Tail15 winner = winner :=
  (root22Step15).trans root22Scan16
theorem root22Scan14 : scan edge (search edge 3) 4 [97] 97 root22Tail14 winner = winner :=
  (root22Step14).trans root22Scan15
theorem root22Scan13 : scan edge (search edge 3) 4 [97] 97 root22Tail13 winner = winner :=
  (root22Step13).trans root22Scan14
theorem root22Scan12 : scan edge (search edge 3) 4 [97] 97 root22Tail12 winner = winner :=
  (root22Step12).trans root22Scan13
theorem root22Scan11 : scan edge (search edge 3) 4 [97] 97 root22Tail11 winner = winner :=
  (root22Step11).trans root22Scan12
theorem root22Scan10 : scan edge (search edge 3) 4 [97] 97 root22Tail10 winner = winner :=
  (root22Step10).trans root22Scan11
theorem root22Scan9 : scan edge (search edge 3) 4 [97] 97 root22Tail9 winner = winner :=
  (root22Step9).trans root22Scan10
theorem root22Scan8 : scan edge (search edge 3) 4 [97] 97 root22Tail8 winner = winner :=
  (root22Step8).trans root22Scan9
theorem root22Scan7 : scan edge (search edge 3) 4 [97] 97 root22Tail7 winner = winner :=
  (root22Step7).trans root22Scan8
theorem root22Scan6 : scan edge (search edge 3) 4 [97] 97 root22Tail6 winner = winner :=
  (root22Step6).trans root22Scan7
theorem root22Scan5 : scan edge (search edge 3) 4 [97] 97 root22Tail5 winner = winner :=
  (root22Step5).trans root22Scan6
theorem root22Scan4 : scan edge (search edge 3) 4 [97] 97 root22Tail4 winner = winner :=
  (root22Step4).trans root22Scan5
theorem root22Scan3 : scan edge (search edge 3) 4 [97] 97 root22Tail3 winner = winner :=
  (root22Step3).trans root22Scan4
theorem root22Scan2 : scan edge (search edge 3) 4 [97] 97 root22Tail2 winner = winner :=
  (root22Step2).trans root22Scan3
theorem root22Scan1 : scan edge (search edge 3) 4 [97] 97 root22Tail1 winner = winner :=
  (root22Step1).trans root22Scan2
theorem root22Scan0 : scan edge (search edge 3) 4 [97] 97 root22Tail0 winner = winner :=
  (root22Step0).trans root22Scan1
set_option maxRecDepth 4096 in
theorem root22Branch : search edge 4 [97] root22Candidates 97 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [97] 97 root22Tail0 winner = winner
  exact root22Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
