import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root295Candidates : List Nat := Data.row1973
set_option maxRecDepth 8192 in
theorem root295Candidates_checked : candidates edge 5 0 1973 tail296 winner.bound = root295Candidates := by
  apply candidates_eq_row edge 5 0 1973 tail296 root295Candidates winner.bound tail296_ordered
  · decide
  · decide
  · funext q
    rfl
def root295Tail50 : List Nat := []
def root295Tail49 : List Nat := 23993 :: root295Tail50
def root295Tail48 : List Nat := 23663 :: root295Tail49
def root295Tail47 : List Nat := 22109 :: root295Tail48
def root295Tail46 : List Nat := 22067 :: root295Tail47
def root295Tail45 : List Nat := 21881 :: root295Tail46
def root295Tail44 : List Nat := 21647 :: root295Tail45
def root295Tail43 : List Nat := 21563 :: root295Tail44
def root295Tail42 : List Nat := 21419 :: root295Tail43
def root295Tail41 : List Nat := 20759 :: root295Tail42
def root295Tail40 : List Nat := 20441 :: root295Tail41
def root295Tail39 : List Nat := 20399 :: root295Tail40
def root295Tail38 : List Nat := 18917 :: root295Tail39
def root295Tail37 : List Nat := 18731 :: root295Tail38
def root295Tail36 : List Nat := 17987 :: root295Tail37
def root295Tail35 : List Nat := 17333 :: root295Tail36
def root295Tail34 : List Nat := 17231 :: root295Tail35
def root295Tail33 : List Nat := 16451 :: root295Tail34
def root295Tail32 : List Nat := 15233 :: root295Tail33
def root295Tail31 : List Nat := 14771 :: root295Tail32
def root295Tail30 : List Nat := 14759 :: root295Tail31
def root295Tail29 : List Nat := 13901 :: root295Tail30
def root295Tail28 : List Nat := 13397 :: root295Tail29
def root295Tail27 : List Nat := 13121 :: root295Tail28
def root295Tail26 : List Nat := 12809 :: root295Tail27
def root295Tail25 : List Nat := 11159 :: root295Tail26
def root295Tail24 : List Nat := 11057 :: root295Tail25
def root295Tail23 : List Nat := 10937 :: root295Tail24
def root295Tail22 : List Nat := 10781 :: root295Tail23
def root295Tail21 : List Nat := 10499 :: root295Tail22
def root295Tail20 : List Nat := 10433 :: root295Tail21
def root295Tail19 : List Nat := 10181 :: root295Tail20
def root295Tail18 : List Nat := 9029 :: root295Tail19
def root295Tail17 : List Nat := 8837 :: root295Tail18
def root295Tail16 : List Nat := 8573 :: root295Tail17
def root295Tail15 : List Nat := 8363 :: root295Tail16
def root295Tail14 : List Nat := 8291 :: root295Tail15
def root295Tail13 : List Nat := 7547 :: root295Tail14
def root295Tail12 : List Nat := 6131 :: root295Tail13
def root295Tail11 : List Nat := 5897 :: root295Tail12
def root295Tail10 : List Nat := 5741 :: root295Tail11
def root295Tail9 : List Nat := 5711 :: root295Tail10
def root295Tail8 : List Nat := 4919 :: root295Tail9
def root295Tail7 : List Nat := 4751 :: root295Tail8
def root295Tail6 : List Nat := 4493 :: root295Tail7
def root295Tail5 : List Nat := 4457 :: root295Tail6
def root295Tail4 : List Nat := 4409 :: root295Tail5
def root295Tail3 : List Nat := 4397 :: root295Tail4
def root295Tail2 : List Nat := 2591 :: root295Tail3
def root295Tail1 : List Nat := 2381 :: root295Tail2
def root295Tail0 : List Nat := 1979 :: root295Tail1
def root295Candidates0 : List Nat := [8291, 18731]
set_option maxRecDepth 8192 in
theorem root295Candidates0_checked : candidates edge 4 1973 1979 root295Tail1 winner.bound = root295Candidates0 := by
  unfold candidates
  have row : edge 1979 = (SortedNatTree.contains FastRows.row1979) := by
    change Data.row1979.contains = (SortedNatTree.contains FastRows.row1979)
    exact FastRows.row1979_checked.symm
  rw [row]
  decide
theorem root295Step0 : scan edge (search edge 3) 4 [1973] 1973 root295Tail0 winner =
    scan edge (search edge 3) 4 [1973] 1973 root295Tail1 winner := by
  rw [root295Tail0, scan, if_neg (by decide), root295Candidates0_checked]
  change scan edge (search edge 3) 4 [1973] 1973 root295Tail1
    (if 3 ≤ root295Candidates0.length then search edge 3 [1973, 1979] root295Candidates0 3952 winner else winner) = _
  rw [if_neg (by decide)]
def root295Candidates1 : List Nat := [10781]
set_option maxRecDepth 8192 in
theorem root295Candidates1_checked : candidates edge 4 1973 2381 root295Tail2 winner.bound = root295Candidates1 := by
  unfold candidates
  have row : edge 2381 = (SortedNatTree.contains FastRows.row2381) := by
    change Data.row2381.contains = (SortedNatTree.contains FastRows.row2381)
    exact FastRows.row2381_checked.symm
  rw [row]
  decide
theorem root295Step1 : scan edge (search edge 3) 4 [1973] 1973 root295Tail1 winner =
    scan edge (search edge 3) 4 [1973] 1973 root295Tail2 winner := by
  rw [root295Tail1, scan, if_neg (by decide), root295Candidates1_checked]
  change scan edge (search edge 3) 4 [1973] 1973 root295Tail2
    (if 3 ≤ root295Candidates1.length then search edge 3 [1973, 2381] root295Candidates1 4354 winner else winner) = _
  rw [if_neg (by decide)]
def root295Candidates2 : List Nat := [5741, 13121, 20759]
set_option maxRecDepth 8192 in
theorem root295Candidates2_checked : candidates edge 4 1973 2591 root295Tail3 winner.bound = root295Candidates2 := by
  unfold candidates
  have row : edge 2591 = (SortedNatTree.contains FastRows.row2591) := by
    change Data.row2591.contains = (SortedNatTree.contains FastRows.row2591)
    exact FastRows.row2591_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root295Child2 : search edge 3 [1973, 2591] root295Candidates2 4564 winner = winner := by decide
theorem root295Step2 : scan edge (search edge 3) 4 [1973] 1973 root295Tail2 winner =
    scan edge (search edge 3) 4 [1973] 1973 root295Tail3 winner := by
  rw [root295Tail2, scan, if_neg (by decide), root295Candidates2_checked]
  change scan edge (search edge 3) 4 [1973] 1973 root295Tail3
    (if 3 ≤ root295Candidates2.length then search edge 3 [1973, 2591] root295Candidates2 4564 winner else winner) = _
  rw [if_pos (by decide), root295Child2]
def root295Candidates3 : List Nat := [11057, 12809, 13121, 16451]
set_option maxRecDepth 8192 in
theorem root295Candidates3_checked : candidates edge 4 1973 4397 root295Tail4 winner.bound = root295Candidates3 := by
  unfold candidates
  have row : edge 4397 = (SortedNatTree.contains FastRows.row4397) := by
    change Data.row4397.contains = (SortedNatTree.contains FastRows.row4397)
    exact FastRows.row4397_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root295Child3 : search edge 3 [1973, 4397] root295Candidates3 6370 winner = winner := by decide
theorem root295Step3 : scan edge (search edge 3) 4 [1973] 1973 root295Tail3 winner =
    scan edge (search edge 3) 4 [1973] 1973 root295Tail4 winner := by
  rw [root295Tail3, scan, if_neg (by decide), root295Candidates3_checked]
  change scan edge (search edge 3) 4 [1973] 1973 root295Tail4
    (if 3 ≤ root295Candidates3.length then search edge 3 [1973, 4397] root295Candidates3 6370 winner else winner) = _
  rw [if_pos (by decide), root295Child3]
def root295Candidates4 : List Nat := [10937]
set_option maxRecDepth 8192 in
theorem root295Candidates4_checked : candidates edge 4 1973 4409 root295Tail5 winner.bound = root295Candidates4 := by
  unfold candidates
  have row : edge 4409 = (SortedNatTree.contains FastRows.row4409) := by
    change Data.row4409.contains = (SortedNatTree.contains FastRows.row4409)
    exact FastRows.row4409_checked.symm
  rw [row]
  decide
theorem root295Step4 : scan edge (search edge 3) 4 [1973] 1973 root295Tail4 winner =
    scan edge (search edge 3) 4 [1973] 1973 root295Tail5 winner := by
  rw [root295Tail4, scan, if_neg (by decide), root295Candidates4_checked]
  change scan edge (search edge 3) 4 [1973] 1973 root295Tail5
    (if 3 ≤ root295Candidates4.length then search edge 3 [1973, 4409] root295Candidates4 6382 winner else winner) = _
  rw [if_neg (by decide)]
def root295Candidates5 : List Nat := [12809]
set_option maxRecDepth 8192 in
theorem root295Candidates5_checked : candidates edge 4 1973 4457 root295Tail6 winner.bound = root295Candidates5 := by
  unfold candidates
  have row : edge 4457 = (SortedNatTree.contains FastRows.row4457) := by
    change Data.row4457.contains = (SortedNatTree.contains FastRows.row4457)
    exact FastRows.row4457_checked.symm
  rw [row]
  decide
theorem root295Step5 : scan edge (search edge 3) 4 [1973] 1973 root295Tail5 winner =
    scan edge (search edge 3) 4 [1973] 1973 root295Tail6 winner := by
  rw [root295Tail5, scan, if_neg (by decide), root295Candidates5_checked]
  change scan edge (search edge 3) 4 [1973] 1973 root295Tail6
    (if 3 ≤ root295Candidates5.length then search edge 3 [1973, 4457] root295Candidates5 6430 winner else winner) = _
  rw [if_neg (by decide)]
def root295Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root295Candidates6_checked : candidates edge 4 1973 4493 root295Tail7 winner.bound = root295Candidates6 := by
  unfold candidates
  have row : edge 4493 = (SortedNatTree.contains FastRows.row4493) := by
    change Data.row4493.contains = (SortedNatTree.contains FastRows.row4493)
    exact FastRows.row4493_checked.symm
  rw [row]
  decide
theorem root295Step6 : scan edge (search edge 3) 4 [1973] 1973 root295Tail6 winner =
    scan edge (search edge 3) 4 [1973] 1973 root295Tail7 winner := by
  rw [root295Tail6, scan, if_neg (by decide), root295Candidates6_checked]
  change scan edge (search edge 3) 4 [1973] 1973 root295Tail7
    (if 3 ≤ root295Candidates6.length then search edge 3 [1973, 4493] root295Candidates6 6466 winner else winner) = _
  rw [if_neg (by decide)]
def root295Candidates7 : List Nat := [5711, 8837, 13901]
set_option maxRecDepth 8192 in
theorem root295Candidates7_checked : candidates edge 4 1973 4751 root295Tail8 winner.bound = root295Candidates7 := by
  unfold candidates
  have row : edge 4751 = (SortedNatTree.contains FastRows.row4751) := by
    change Data.row4751.contains = (SortedNatTree.contains FastRows.row4751)
    exact FastRows.row4751_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root295Child7 : search edge 3 [1973, 4751] root295Candidates7 6724 winner = winner := by decide
theorem root295Step7 : scan edge (search edge 3) 4 [1973] 1973 root295Tail7 winner =
    scan edge (search edge 3) 4 [1973] 1973 root295Tail8 winner := by
  rw [root295Tail7, scan, if_neg (by decide), root295Candidates7_checked]
  change scan edge (search edge 3) 4 [1973] 1973 root295Tail8
    (if 3 ≤ root295Candidates7.length then search edge 3 [1973, 4751] root295Candidates7 6724 winner else winner) = _
  rw [if_pos (by decide), root295Child7]
def root295Candidates8 : List Nat := [5897, 10781]
set_option maxRecDepth 8192 in
theorem root295Candidates8_checked : candidates edge 4 1973 4919 root295Tail9 winner.bound = root295Candidates8 := by
  unfold candidates
  have row : edge 4919 = (SortedNatTree.contains FastRows.row4919) := by
    change Data.row4919.contains = (SortedNatTree.contains FastRows.row4919)
    exact FastRows.row4919_checked.symm
  rw [row]
  decide
theorem root295Step8 : scan edge (search edge 3) 4 [1973] 1973 root295Tail8 winner =
    scan edge (search edge 3) 4 [1973] 1973 root295Tail9 winner := by
  rw [root295Tail8, scan, if_neg (by decide), root295Candidates8_checked]
  change scan edge (search edge 3) 4 [1973] 1973 root295Tail9
    (if 3 ≤ root295Candidates8.length then search edge 3 [1973, 4919] root295Candidates8 6892 winner else winner) = _
  rw [if_neg (by decide)]
def root295Candidates9 : List Nat := [8291, 8573, 10781, 14759, 17231]
set_option maxRecDepth 8192 in
theorem root295Candidates9_checked : candidates edge 4 1973 5711 root295Tail10 winner.bound = root295Candidates9 := by
  unfold candidates
  have row : edge 5711 = (SortedNatTree.contains FastRows.row5711) := by
    change Data.row5711.contains = (SortedNatTree.contains FastRows.row5711)
    exact FastRows.row5711_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root295Child9 : search edge 3 [1973, 5711] root295Candidates9 7684 winner = winner := by decide
theorem root295Step9 : scan edge (search edge 3) 4 [1973] 1973 root295Tail9 winner =
    scan edge (search edge 3) 4 [1973] 1973 root295Tail10 winner := by
  rw [root295Tail9, scan, if_neg (by decide), root295Candidates9_checked]
  change scan edge (search edge 3) 4 [1973] 1973 root295Tail10
    (if 3 ≤ root295Candidates9.length then search edge 3 [1973, 5711] root295Candidates9 7684 winner else winner) = _
  rw [if_pos (by decide), root295Child9]
def root295Candidates10 : List Nat := [12809, 17987]
set_option maxRecDepth 8192 in
theorem root295Candidates10_checked : candidates edge 4 1973 5741 root295Tail11 winner.bound = root295Candidates10 := by
  unfold candidates
  have row : edge 5741 = (SortedNatTree.contains FastRows.row5741) := by
    change Data.row5741.contains = (SortedNatTree.contains FastRows.row5741)
    exact FastRows.row5741_checked.symm
  rw [row]
  decide
theorem root295Step10 : scan edge (search edge 3) 4 [1973] 1973 root295Tail10 winner =
    scan edge (search edge 3) 4 [1973] 1973 root295Tail11 winner := by
  rw [root295Tail10, scan, if_neg (by decide), root295Candidates10_checked]
  change scan edge (search edge 3) 4 [1973] 1973 root295Tail11
    (if 3 ≤ root295Candidates10.length then search edge 3 [1973, 5741] root295Candidates10 7714 winner else winner) = _
  rw [if_neg (by decide)]
def root295Candidates11 : List Nat := [6131, 10937]
set_option maxRecDepth 8192 in
theorem root295Candidates11_checked : candidates edge 4 1973 5897 root295Tail12 winner.bound = root295Candidates11 := by
  unfold candidates
  have row : edge 5897 = (SortedNatTree.contains FastRows.row5897) := by
    change Data.row5897.contains = (SortedNatTree.contains FastRows.row5897)
    exact FastRows.row5897_checked.symm
  rw [row]
  decide
theorem root295Step11 : scan edge (search edge 3) 4 [1973] 1973 root295Tail11 winner =
    scan edge (search edge 3) 4 [1973] 1973 root295Tail12 winner := by
  rw [root295Tail11, scan, if_neg (by decide), root295Candidates11_checked]
  change scan edge (search edge 3) 4 [1973] 1973 root295Tail12
    (if 3 ≤ root295Candidates11.length then search edge 3 [1973, 5897] root295Candidates11 7870 winner else winner) = _
  rw [if_neg (by decide)]
theorem root295Scan12 : scan edge (search edge 3) 4 [1973] 1973 root295Tail12 winner = winner := by rw [root295Tail12, scan, if_pos (by decide)]
theorem root295Scan11 : scan edge (search edge 3) 4 [1973] 1973 root295Tail11 winner = winner :=
  (root295Step11).trans root295Scan12
theorem root295Scan10 : scan edge (search edge 3) 4 [1973] 1973 root295Tail10 winner = winner :=
  (root295Step10).trans root295Scan11
theorem root295Scan9 : scan edge (search edge 3) 4 [1973] 1973 root295Tail9 winner = winner :=
  (root295Step9).trans root295Scan10
theorem root295Scan8 : scan edge (search edge 3) 4 [1973] 1973 root295Tail8 winner = winner :=
  (root295Step8).trans root295Scan9
theorem root295Scan7 : scan edge (search edge 3) 4 [1973] 1973 root295Tail7 winner = winner :=
  (root295Step7).trans root295Scan8
theorem root295Scan6 : scan edge (search edge 3) 4 [1973] 1973 root295Tail6 winner = winner :=
  (root295Step6).trans root295Scan7
theorem root295Scan5 : scan edge (search edge 3) 4 [1973] 1973 root295Tail5 winner = winner :=
  (root295Step5).trans root295Scan6
theorem root295Scan4 : scan edge (search edge 3) 4 [1973] 1973 root295Tail4 winner = winner :=
  (root295Step4).trans root295Scan5
theorem root295Scan3 : scan edge (search edge 3) 4 [1973] 1973 root295Tail3 winner = winner :=
  (root295Step3).trans root295Scan4
theorem root295Scan2 : scan edge (search edge 3) 4 [1973] 1973 root295Tail2 winner = winner :=
  (root295Step2).trans root295Scan3
theorem root295Scan1 : scan edge (search edge 3) 4 [1973] 1973 root295Tail1 winner = winner :=
  (root295Step1).trans root295Scan2
theorem root295Scan0 : scan edge (search edge 3) 4 [1973] 1973 root295Tail0 winner = winner :=
  (root295Step0).trans root295Scan1
set_option maxRecDepth 4096 in
theorem root295Branch : search edge 4 [1973] root295Candidates 1973 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1973] 1973 root295Tail0 winner = winner
  exact root295Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
