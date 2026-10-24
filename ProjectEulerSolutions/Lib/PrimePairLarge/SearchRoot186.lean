import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root186Candidates : List Nat := Data.row1129
set_option maxRecDepth 8192 in
theorem root186Candidates_checked : candidates edge 5 0 1129 tail187 winner.bound = root186Candidates := by
  apply candidates_eq_row edge 5 0 1129 tail187 root186Candidates winner.bound tail187_ordered
  · decide
  · decide
  · funext q
    rfl
def root186Tail51 : List Nat := []
def root186Tail50 : List Nat := 23053 :: root186Tail51
def root186Tail49 : List Nat := 23011 :: root186Tail50
def root186Tail48 : List Nat := 22189 :: root186Tail49
def root186Tail47 : List Nat := 21799 :: root186Tail48
def root186Tail46 : List Nat := 21649 :: root186Tail47
def root186Tail45 : List Nat := 21517 :: root186Tail46
def root186Tail44 : List Nat := 21163 :: root186Tail45
def root186Tail43 : List Nat := 19759 :: root186Tail44
def root186Tail42 : List Nat := 19381 :: root186Tail43
def root186Tail41 : List Nat := 19237 :: root186Tail42
def root186Tail40 : List Nat := 18433 :: root186Tail41
def root186Tail39 : List Nat := 17137 :: root186Tail40
def root186Tail38 : List Nat := 17107 :: root186Tail39
def root186Tail37 : List Nat := 16651 :: root186Tail38
def root186Tail36 : List Nat := 16561 :: root186Tail37
def root186Tail35 : List Nat := 16189 :: root186Tail36
def root186Tail34 : List Nat := 15643 :: root186Tail35
def root186Tail33 : List Nat := 15259 :: root186Tail34
def root186Tail32 : List Nat := 13477 :: root186Tail33
def root186Tail31 : List Nat := 13309 :: root186Tail32
def root186Tail30 : List Nat := 12097 :: root186Tail31
def root186Tail29 : List Nat := 11047 :: root186Tail30
def root186Tail28 : List Nat := 10429 :: root186Tail29
def root186Tail27 : List Nat := 10159 :: root186Tail28
def root186Tail26 : List Nat := 9931 :: root186Tail27
def root186Tail25 : List Nat := 9697 :: root186Tail26
def root186Tail24 : List Nat := 9511 :: root186Tail25
def root186Tail23 : List Nat := 8737 :: root186Tail24
def root186Tail22 : List Nat := 8731 :: root186Tail23
def root186Tail21 : List Nat := 8629 :: root186Tail22
def root186Tail20 : List Nat := 8233 :: root186Tail21
def root186Tail19 : List Nat := 7621 :: root186Tail20
def root186Tail18 : List Nat := 7603 :: root186Tail19
def root186Tail17 : List Nat := 7417 :: root186Tail18
def root186Tail16 : List Nat := 7057 :: root186Tail17
def root186Tail15 : List Nat := 6871 :: root186Tail16
def root186Tail14 : List Nat := 6829 :: root186Tail15
def root186Tail13 : List Nat := 6451 :: root186Tail14
def root186Tail12 : List Nat := 6421 :: root186Tail13
def root186Tail11 : List Nat := 5923 :: root186Tail12
def root186Tail10 : List Nat := 5107 :: root186Tail11
def root186Tail9 : List Nat := 4561 :: root186Tail10
def root186Tail8 : List Nat := 3001 :: root186Tail9
def root186Tail7 : List Nat := 2767 :: root186Tail8
def root186Tail6 : List Nat := 2647 :: root186Tail7
def root186Tail5 : List Nat := 2557 :: root186Tail6
def root186Tail4 : List Nat := 2473 :: root186Tail5
def root186Tail3 : List Nat := 2269 :: root186Tail4
def root186Tail2 : List Nat := 2143 :: root186Tail3
def root186Tail1 : List Nat := 1999 :: root186Tail2
def root186Tail0 : List Nat := 1741 :: root186Tail1
def root186Candidates0 : List Nat := [10159, 13309]
set_option maxRecDepth 8192 in
theorem root186Candidates0_checked : candidates edge 4 1129 1741 root186Tail1 winner.bound = root186Candidates0 := by
  unfold candidates
  have row : edge 1741 = (SortedNatTree.contains FastRows.row1741) := by
    change Data.row1741.contains = (SortedNatTree.contains FastRows.row1741)
    exact FastRows.row1741_checked.symm
  rw [row]
  decide
theorem root186Step0 : scan edge (search edge 3) 4 [1129] 1129 root186Tail0 winner =
    scan edge (search edge 3) 4 [1129] 1129 root186Tail1 winner := by
  rw [root186Tail0, scan, if_neg (by decide), root186Candidates0_checked]
  change scan edge (search edge 3) 4 [1129] 1129 root186Tail1
    (if 3 ≤ root186Candidates0.length then search edge 3 [1129, 1741] root186Candidates0 2870 winner else winner) = _
  rw [if_neg (by decide)]
def root186Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root186Candidates1_checked : candidates edge 4 1129 1999 root186Tail2 winner.bound = root186Candidates1 := by
  unfold candidates
  have row : edge 1999 = (SortedNatTree.contains FastRows.row1999) := by
    change Data.row1999.contains = (SortedNatTree.contains FastRows.row1999)
    exact FastRows.row1999_checked.symm
  rw [row]
  decide
theorem root186Step1 : scan edge (search edge 3) 4 [1129] 1129 root186Tail1 winner =
    scan edge (search edge 3) 4 [1129] 1129 root186Tail2 winner := by
  rw [root186Tail1, scan, if_neg (by decide), root186Candidates1_checked]
  change scan edge (search edge 3) 4 [1129] 1129 root186Tail2
    (if 3 ≤ root186Candidates1.length then search edge 3 [1129, 1999] root186Candidates1 3128 winner else winner) = _
  rw [if_neg (by decide)]
def root186Candidates2 : List Nat := [6871, 17137]
set_option maxRecDepth 8192 in
theorem root186Candidates2_checked : candidates edge 4 1129 2143 root186Tail3 winner.bound = root186Candidates2 := by
  unfold candidates
  have row : edge 2143 = (SortedNatTree.contains FastRows.row2143) := by
    change Data.row2143.contains = (SortedNatTree.contains FastRows.row2143)
    exact FastRows.row2143_checked.symm
  rw [row]
  decide
theorem root186Step2 : scan edge (search edge 3) 4 [1129] 1129 root186Tail2 winner =
    scan edge (search edge 3) 4 [1129] 1129 root186Tail3 winner := by
  rw [root186Tail2, scan, if_neg (by decide), root186Candidates2_checked]
  change scan edge (search edge 3) 4 [1129] 1129 root186Tail3
    (if 3 ≤ root186Candidates2.length then search edge 3 [1129, 2143] root186Candidates2 3272 winner else winner) = _
  rw [if_neg (by decide)]
def root186Candidates3 : List Nat := [5107, 5923, 9697, 10429]
set_option maxRecDepth 8192 in
theorem root186Candidates3_checked : candidates edge 4 1129 2269 root186Tail4 winner.bound = root186Candidates3 := by
  unfold candidates
  have row : edge 2269 = (SortedNatTree.contains FastRows.row2269) := by
    change Data.row2269.contains = (SortedNatTree.contains FastRows.row2269)
    exact FastRows.row2269_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root186Child3 : search edge 3 [1129, 2269] root186Candidates3 3398 winner = winner := by decide
theorem root186Step3 : scan edge (search edge 3) 4 [1129] 1129 root186Tail3 winner =
    scan edge (search edge 3) 4 [1129] 1129 root186Tail4 winner := by
  rw [root186Tail3, scan, if_neg (by decide), root186Candidates3_checked]
  change scan edge (search edge 3) 4 [1129] 1129 root186Tail4
    (if 3 ≤ root186Candidates3.length then search edge 3 [1129, 2269] root186Candidates3 3398 winner else winner) = _
  rw [if_pos (by decide), root186Child3]
def root186Candidates4 : List Nat := [2647, 3001, 7417, 19381]
set_option maxRecDepth 8192 in
theorem root186Candidates4_checked : candidates edge 4 1129 2473 root186Tail5 winner.bound = root186Candidates4 := by
  unfold candidates
  have row : edge 2473 = (SortedNatTree.contains FastRows.row2473) := by
    change Data.row2473.contains = (SortedNatTree.contains FastRows.row2473)
    exact FastRows.row2473_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root186Child4 : search edge 3 [1129, 2473] root186Candidates4 3602 winner = winner := by decide
theorem root186Step4 : scan edge (search edge 3) 4 [1129] 1129 root186Tail4 winner =
    scan edge (search edge 3) 4 [1129] 1129 root186Tail5 winner := by
  rw [root186Tail4, scan, if_neg (by decide), root186Candidates4_checked]
  change scan edge (search edge 3) 4 [1129] 1129 root186Tail5
    (if 3 ≤ root186Candidates4.length then search edge 3 [1129, 2473] root186Candidates4 3602 winner else winner) = _
  rw [if_pos (by decide), root186Child4]
def root186Candidates5 : List Nat := [6871, 12097, 13477]
set_option maxRecDepth 8192 in
theorem root186Candidates5_checked : candidates edge 4 1129 2557 root186Tail6 winner.bound = root186Candidates5 := by
  unfold candidates
  have row : edge 2557 = (SortedNatTree.contains FastRows.row2557) := by
    change Data.row2557.contains = (SortedNatTree.contains FastRows.row2557)
    exact FastRows.row2557_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root186Child5 : search edge 3 [1129, 2557] root186Candidates5 3686 winner = winner := by decide
theorem root186Step5 : scan edge (search edge 3) 4 [1129] 1129 root186Tail5 winner =
    scan edge (search edge 3) 4 [1129] 1129 root186Tail6 winner := by
  rw [root186Tail5, scan, if_neg (by decide), root186Candidates5_checked]
  change scan edge (search edge 3) 4 [1129] 1129 root186Tail6
    (if 3 ≤ root186Candidates5.length then search edge 3 [1129, 2557] root186Candidates5 3686 winner else winner) = _
  rw [if_pos (by decide), root186Child5]
def root186Candidates6 : List Nat := [8737, 15259]
set_option maxRecDepth 8192 in
theorem root186Candidates6_checked : candidates edge 4 1129 2647 root186Tail7 winner.bound = root186Candidates6 := by
  unfold candidates
  have row : edge 2647 = (SortedNatTree.contains FastRows.row2647) := by
    change Data.row2647.contains = (SortedNatTree.contains FastRows.row2647)
    exact FastRows.row2647_checked.symm
  rw [row]
  decide
theorem root186Step6 : scan edge (search edge 3) 4 [1129] 1129 root186Tail6 winner =
    scan edge (search edge 3) 4 [1129] 1129 root186Tail7 winner := by
  rw [root186Tail6, scan, if_neg (by decide), root186Candidates6_checked]
  change scan edge (search edge 3) 4 [1129] 1129 root186Tail7
    (if 3 ≤ root186Candidates6.length then search edge 3 [1129, 2647] root186Candidates6 3776 winner else winner) = _
  rw [if_neg (by decide)]
def root186Candidates7 : List Nat := [5107, 7621, 18433]
set_option maxRecDepth 8192 in
theorem root186Candidates7_checked : candidates edge 4 1129 2767 root186Tail8 winner.bound = root186Candidates7 := by
  unfold candidates
  have row : edge 2767 = (SortedNatTree.contains FastRows.row2767) := by
    change Data.row2767.contains = (SortedNatTree.contains FastRows.row2767)
    exact FastRows.row2767_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root186Child7 : search edge 3 [1129, 2767] root186Candidates7 3896 winner = winner := by decide
theorem root186Step7 : scan edge (search edge 3) 4 [1129] 1129 root186Tail7 winner =
    scan edge (search edge 3) 4 [1129] 1129 root186Tail8 winner := by
  rw [root186Tail7, scan, if_neg (by decide), root186Candidates7_checked]
  change scan edge (search edge 3) 4 [1129] 1129 root186Tail8
    (if 3 ≤ root186Candidates7.length then search edge 3 [1129, 2767] root186Candidates7 3896 winner else winner) = _
  rw [if_pos (by decide), root186Child7]
def root186Candidates8 : List Nat := [7621, 9697, 9931, 19381]
set_option maxRecDepth 8192 in
theorem root186Candidates8_checked : candidates edge 4 1129 3001 root186Tail9 winner.bound = root186Candidates8 := by
  unfold candidates
  have row : edge 3001 = (SortedNatTree.contains FastRows.row3001) := by
    change Data.row3001.contains = (SortedNatTree.contains FastRows.row3001)
    exact FastRows.row3001_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root186Child8 : search edge 3 [1129, 3001] root186Candidates8 4130 winner = winner := by decide
theorem root186Step8 : scan edge (search edge 3) 4 [1129] 1129 root186Tail8 winner =
    scan edge (search edge 3) 4 [1129] 1129 root186Tail9 winner := by
  rw [root186Tail8, scan, if_neg (by decide), root186Candidates8_checked]
  change scan edge (search edge 3) 4 [1129] 1129 root186Tail9
    (if 3 ≤ root186Candidates8.length then search edge 3 [1129, 3001] root186Candidates8 4130 winner else winner) = _
  rw [if_pos (by decide), root186Child8]
def root186Candidates9 : List Nat := [8737, 13309]
set_option maxRecDepth 8192 in
theorem root186Candidates9_checked : candidates edge 4 1129 4561 root186Tail10 winner.bound = root186Candidates9 := by
  unfold candidates
  have row : edge 4561 = (SortedNatTree.contains FastRows.row4561) := by
    change Data.row4561.contains = (SortedNatTree.contains FastRows.row4561)
    exact FastRows.row4561_checked.symm
  rw [row]
  decide
theorem root186Step9 : scan edge (search edge 3) 4 [1129] 1129 root186Tail9 winner =
    scan edge (search edge 3) 4 [1129] 1129 root186Tail10 winner := by
  rw [root186Tail9, scan, if_neg (by decide), root186Candidates9_checked]
  change scan edge (search edge 3) 4 [1129] 1129 root186Tail10
    (if 3 ≤ root186Candidates9.length then search edge 3 [1129, 4561] root186Candidates9 5690 winner else winner) = _
  rw [if_neg (by decide)]
def root186Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root186Candidates10_checked : candidates edge 4 1129 5107 root186Tail11 winner.bound = root186Candidates10 := by
  unfold candidates
  have row : edge 5107 = (SortedNatTree.contains FastRows.row5107) := by
    change Data.row5107.contains = (SortedNatTree.contains FastRows.row5107)
    exact FastRows.row5107_checked.symm
  rw [row]
  decide
theorem root186Step10 : scan edge (search edge 3) 4 [1129] 1129 root186Tail10 winner =
    scan edge (search edge 3) 4 [1129] 1129 root186Tail11 winner := by
  rw [root186Tail10, scan, if_neg (by decide), root186Candidates10_checked]
  change scan edge (search edge 3) 4 [1129] 1129 root186Tail11
    (if 3 ≤ root186Candidates10.length then search edge 3 [1129, 5107] root186Candidates10 6236 winner else winner) = _
  rw [if_neg (by decide)]
def root186Candidates11 : List Nat := [6421, 6871, 7417]
set_option maxRecDepth 8192 in
theorem root186Candidates11_checked : candidates edge 4 1129 5923 root186Tail12 winner.bound = root186Candidates11 := by
  unfold candidates
  have row : edge 5923 = (SortedNatTree.contains FastRows.row5923) := by
    change Data.row5923.contains = (SortedNatTree.contains FastRows.row5923)
    exact FastRows.row5923_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root186Child11 : search edge 3 [1129, 5923] root186Candidates11 7052 winner = winner := by decide
theorem root186Step11 : scan edge (search edge 3) 4 [1129] 1129 root186Tail11 winner =
    scan edge (search edge 3) 4 [1129] 1129 root186Tail12 winner := by
  rw [root186Tail11, scan, if_neg (by decide), root186Candidates11_checked]
  change scan edge (search edge 3) 4 [1129] 1129 root186Tail12
    (if 3 ≤ root186Candidates11.length then search edge 3 [1129, 5923] root186Candidates11 7052 winner else winner) = _
  rw [if_pos (by decide), root186Child11]
theorem root186Scan12 : scan edge (search edge 3) 4 [1129] 1129 root186Tail12 winner = winner := by rw [root186Tail12, scan, if_pos (by decide)]
theorem root186Scan11 : scan edge (search edge 3) 4 [1129] 1129 root186Tail11 winner = winner :=
  (root186Step11).trans root186Scan12
theorem root186Scan10 : scan edge (search edge 3) 4 [1129] 1129 root186Tail10 winner = winner :=
  (root186Step10).trans root186Scan11
theorem root186Scan9 : scan edge (search edge 3) 4 [1129] 1129 root186Tail9 winner = winner :=
  (root186Step9).trans root186Scan10
theorem root186Scan8 : scan edge (search edge 3) 4 [1129] 1129 root186Tail8 winner = winner :=
  (root186Step8).trans root186Scan9
theorem root186Scan7 : scan edge (search edge 3) 4 [1129] 1129 root186Tail7 winner = winner :=
  (root186Step7).trans root186Scan8
theorem root186Scan6 : scan edge (search edge 3) 4 [1129] 1129 root186Tail6 winner = winner :=
  (root186Step6).trans root186Scan7
theorem root186Scan5 : scan edge (search edge 3) 4 [1129] 1129 root186Tail5 winner = winner :=
  (root186Step5).trans root186Scan6
theorem root186Scan4 : scan edge (search edge 3) 4 [1129] 1129 root186Tail4 winner = winner :=
  (root186Step4).trans root186Scan5
theorem root186Scan3 : scan edge (search edge 3) 4 [1129] 1129 root186Tail3 winner = winner :=
  (root186Step3).trans root186Scan4
theorem root186Scan2 : scan edge (search edge 3) 4 [1129] 1129 root186Tail2 winner = winner :=
  (root186Step2).trans root186Scan3
theorem root186Scan1 : scan edge (search edge 3) 4 [1129] 1129 root186Tail1 winner = winner :=
  (root186Step1).trans root186Scan2
theorem root186Scan0 : scan edge (search edge 3) 4 [1129] 1129 root186Tail0 winner = winner :=
  (root186Step0).trans root186Scan1
set_option maxRecDepth 4096 in
theorem root186Branch : search edge 4 [1129] root186Candidates 1129 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1129] 1129 root186Tail0 winner = winner
  exact root186Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
