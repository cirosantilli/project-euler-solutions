import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot003Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root3Candidates0 : List Nat := [577, 709, 997, 1861, 2113, 5077, 6043, 6991, 7507, 8179, 8599, 9697, 9721, 12277, 13999, 19273, 20173, 23893]
set_option maxRecDepth 8192 in
theorem root3Candidates0_checked : candidates edge 4 13 19 root3Tail1 seed.bound = root3Candidates0 := by
  unfold candidates
  have row : edge 19 = (SortedNatTree.contains FastRows.row19) := by
    change Data.row19.contains = (SortedNatTree.contains FastRows.row19)
    exact FastRows.row19_checked.symm
  rw [row]
  decide
def root3Child0Tail18 : List Nat := []
def root3Child0Tail17 : List Nat := 23893 :: root3Child0Tail18
def root3Child0Tail16 : List Nat := 20173 :: root3Child0Tail17
def root3Child0Tail15 : List Nat := 19273 :: root3Child0Tail16
def root3Child0Tail14 : List Nat := 13999 :: root3Child0Tail15
def root3Child0Tail13 : List Nat := 12277 :: root3Child0Tail14
def root3Child0Tail12 : List Nat := 9721 :: root3Child0Tail13
def root3Child0Tail11 : List Nat := 9697 :: root3Child0Tail12
def root3Child0Tail10 : List Nat := 8599 :: root3Child0Tail11
def root3Child0Tail9 : List Nat := 8179 :: root3Child0Tail10
def root3Child0Tail8 : List Nat := 7507 :: root3Child0Tail9
def root3Child0Tail7 : List Nat := 6991 :: root3Child0Tail8
def root3Child0Tail6 : List Nat := 6043 :: root3Child0Tail7
def root3Child0Tail5 : List Nat := 5077 :: root3Child0Tail6
def root3Child0Tail4 : List Nat := 2113 :: root3Child0Tail5
def root3Child0Tail3 : List Nat := 1861 :: root3Child0Tail4
def root3Child0Tail2 : List Nat := 997 :: root3Child0Tail3
def root3Child0Tail1 : List Nat := 709 :: root3Child0Tail2
def root3Child0Tail0 : List Nat := 577 :: root3Child0Tail1
def root3Child0Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root3Child0Candidates0_checked : candidates edge 3 32 577 root3Child0Tail1 seed.bound = root3Child0Candidates0 := by
  unfold candidates
  have row : edge 577 = (SortedNatTree.contains FastRows.row577) := by
    change Data.row577.contains = (SortedNatTree.contains FastRows.row577)
    exact FastRows.row577_checked.symm
  rw [row]
  decide
def root3Child0Candidates1 : List Nat := [20173]
set_option maxRecDepth 8192 in
theorem root3Child0Candidates1_checked : candidates edge 3 32 709 root3Child0Tail2 seed.bound = root3Child0Candidates1 := by
  unfold candidates
  have row : edge 709 = (SortedNatTree.contains FastRows.row709) := by
    change Data.row709.contains = (SortedNatTree.contains FastRows.row709)
    exact FastRows.row709_checked.symm
  rw [row]
  decide
def root3Child0Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root3Child0Candidates2_checked : candidates edge 3 32 997 root3Child0Tail3 seed.bound = root3Child0Candidates2 := by
  unfold candidates
  have row : edge 997 = (SortedNatTree.contains FastRows.row997) := by
    change Data.row997.contains = (SortedNatTree.contains FastRows.row997)
    exact FastRows.row997_checked.symm
  rw [row]
  decide
def root3Child0Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root3Child0Candidates3_checked : candidates edge 3 32 1861 root3Child0Tail4 seed.bound = root3Child0Candidates3 := by
  unfold candidates
  have row : edge 1861 = (SortedNatTree.contains FastRows.row1861) := by
    change Data.row1861.contains = (SortedNatTree.contains FastRows.row1861)
    exact FastRows.row1861_checked.symm
  rw [row]
  decide
def root3Child0Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root3Child0Candidates4_checked : candidates edge 3 32 2113 root3Child0Tail5 seed.bound = root3Child0Candidates4 := by
  unfold candidates
  have row : edge 2113 = (SortedNatTree.contains FastRows.row2113) := by
    change Data.row2113.contains = (SortedNatTree.contains FastRows.row2113)
    exact FastRows.row2113_checked.symm
  rw [row]
  decide
def root3Child0Candidates5 : List Nat := [6043]
set_option maxRecDepth 8192 in
theorem root3Child0Candidates5_checked : candidates edge 3 32 5077 root3Child0Tail6 seed.bound = root3Child0Candidates5 := by
  unfold candidates
  have row : edge 5077 = (SortedNatTree.contains FastRows.row5077) := by
    change Data.row5077.contains = (SortedNatTree.contains FastRows.row5077)
    exact FastRows.row5077_checked.symm
  rw [row]
  decide
def root3Child0Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root3Child0Candidates6_checked : candidates edge 3 32 6043 root3Child0Tail7 seed.bound = root3Child0Candidates6 := by
  unfold candidates
  have row : edge 6043 = (SortedNatTree.contains FastRows.row6043) := by
    change Data.row6043.contains = (SortedNatTree.contains FastRows.row6043)
    exact FastRows.row6043_checked.symm
  rw [row]
  decide
def root3Child0Candidates7 : List Nat := [9697]
set_option maxRecDepth 8192 in
theorem root3Child0Candidates7_checked : candidates edge 3 32 6991 root3Child0Tail8 seed.bound = root3Child0Candidates7 := by
  unfold candidates
  have row : edge 6991 = (SortedNatTree.contains FastRows.row6991) := by
    change Data.row6991.contains = (SortedNatTree.contains FastRows.row6991)
    exact FastRows.row6991_checked.symm
  rw [row]
  decide
def root3Child0Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root3Child0Candidates8_checked : candidates edge 3 32 7507 root3Child0Tail9 seed.bound = root3Child0Candidates8 := by
  unfold candidates
  have row : edge 7507 = (SortedNatTree.contains FastRows.row7507) := by
    change Data.row7507.contains = (SortedNatTree.contains FastRows.row7507)
    exact FastRows.row7507_checked.symm
  rw [row]
  decide
def root3Child0Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root3Child0Candidates9_checked : candidates edge 3 32 8179 root3Child0Tail10 seed.bound = root3Child0Candidates9 := by
  unfold candidates
  have row : edge 8179 = (SortedNatTree.contains FastRows.row8179) := by
    change Data.row8179.contains = (SortedNatTree.contains FastRows.row8179)
    exact FastRows.row8179_checked.symm
  rw [row]
  decide
def root3Child0Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root3Child0Candidates10_checked : candidates edge 3 32 8599 root3Child0Tail11 seed.bound = root3Child0Candidates10 := by
  unfold candidates
  have row : edge 8599 = (SortedNatTree.contains FastRows.row8599) := by
    change Data.row8599.contains = (SortedNatTree.contains FastRows.row8599)
    exact FastRows.row8599_checked.symm
  rw [row]
  decide
theorem root3Child0Scan11 : scan edge (search edge 2) 3 [13, 19] 32 root3Child0Tail11 seed = seed := by rw [root3Child0Tail11, scan, if_pos (by decide)]
theorem root3Child0Scan10 : scan edge (search edge 2) 3 [13, 19] 32 root3Child0Tail10 seed = seed := by
  rw [root3Child0Tail10, scan, if_neg (by decide), root3Child0Candidates10_checked]
  change scan edge (search edge 2) 3 [13, 19] 32 root3Child0Tail11
    (if 2 ≤ root3Child0Candidates10.length then search edge 2 [13, 19, 8599] root3Child0Candidates10 8631 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root3Child0Scan11
theorem root3Child0Scan9 : scan edge (search edge 2) 3 [13, 19] 32 root3Child0Tail9 seed = seed := by
  rw [root3Child0Tail9, scan, if_neg (by decide), root3Child0Candidates9_checked]
  change scan edge (search edge 2) 3 [13, 19] 32 root3Child0Tail10
    (if 2 ≤ root3Child0Candidates9.length then search edge 2 [13, 19, 8179] root3Child0Candidates9 8211 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root3Child0Scan10
theorem root3Child0Scan8 : scan edge (search edge 2) 3 [13, 19] 32 root3Child0Tail8 seed = seed := by
  rw [root3Child0Tail8, scan, if_neg (by decide), root3Child0Candidates8_checked]
  change scan edge (search edge 2) 3 [13, 19] 32 root3Child0Tail9
    (if 2 ≤ root3Child0Candidates8.length then search edge 2 [13, 19, 7507] root3Child0Candidates8 7539 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root3Child0Scan9
theorem root3Child0Scan7 : scan edge (search edge 2) 3 [13, 19] 32 root3Child0Tail7 seed = seed := by
  rw [root3Child0Tail7, scan, if_neg (by decide), root3Child0Candidates7_checked]
  change scan edge (search edge 2) 3 [13, 19] 32 root3Child0Tail8
    (if 2 ≤ root3Child0Candidates7.length then search edge 2 [13, 19, 6991] root3Child0Candidates7 7023 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root3Child0Scan8
theorem root3Child0Scan6 : scan edge (search edge 2) 3 [13, 19] 32 root3Child0Tail6 seed = seed := by
  rw [root3Child0Tail6, scan, if_neg (by decide), root3Child0Candidates6_checked]
  change scan edge (search edge 2) 3 [13, 19] 32 root3Child0Tail7
    (if 2 ≤ root3Child0Candidates6.length then search edge 2 [13, 19, 6043] root3Child0Candidates6 6075 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root3Child0Scan7
theorem root3Child0Scan5 : scan edge (search edge 2) 3 [13, 19] 32 root3Child0Tail5 seed = seed := by
  rw [root3Child0Tail5, scan, if_neg (by decide), root3Child0Candidates5_checked]
  change scan edge (search edge 2) 3 [13, 19] 32 root3Child0Tail6
    (if 2 ≤ root3Child0Candidates5.length then search edge 2 [13, 19, 5077] root3Child0Candidates5 5109 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root3Child0Scan6
theorem root3Child0Scan4 : scan edge (search edge 2) 3 [13, 19] 32 root3Child0Tail4 seed = seed := by
  rw [root3Child0Tail4, scan, if_neg (by decide), root3Child0Candidates4_checked]
  change scan edge (search edge 2) 3 [13, 19] 32 root3Child0Tail5
    (if 2 ≤ root3Child0Candidates4.length then search edge 2 [13, 19, 2113] root3Child0Candidates4 2145 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root3Child0Scan5
theorem root3Child0Scan3 : scan edge (search edge 2) 3 [13, 19] 32 root3Child0Tail3 seed = seed := by
  rw [root3Child0Tail3, scan, if_neg (by decide), root3Child0Candidates3_checked]
  change scan edge (search edge 2) 3 [13, 19] 32 root3Child0Tail4
    (if 2 ≤ root3Child0Candidates3.length then search edge 2 [13, 19, 1861] root3Child0Candidates3 1893 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root3Child0Scan4
theorem root3Child0Scan2 : scan edge (search edge 2) 3 [13, 19] 32 root3Child0Tail2 seed = seed := by
  rw [root3Child0Tail2, scan, if_neg (by decide), root3Child0Candidates2_checked]
  change scan edge (search edge 2) 3 [13, 19] 32 root3Child0Tail3
    (if 2 ≤ root3Child0Candidates2.length then search edge 2 [13, 19, 997] root3Child0Candidates2 1029 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root3Child0Scan3
theorem root3Child0Scan1 : scan edge (search edge 2) 3 [13, 19] 32 root3Child0Tail1 seed = seed := by
  rw [root3Child0Tail1, scan, if_neg (by decide), root3Child0Candidates1_checked]
  change scan edge (search edge 2) 3 [13, 19] 32 root3Child0Tail2
    (if 2 ≤ root3Child0Candidates1.length then search edge 2 [13, 19, 709] root3Child0Candidates1 741 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root3Child0Scan2
theorem root3Child0Scan0 : scan edge (search edge 2) 3 [13, 19] 32 root3Child0Tail0 seed = seed := by
  rw [root3Child0Tail0, scan, if_neg (by decide), root3Child0Candidates0_checked]
  change scan edge (search edge 2) 3 [13, 19] 32 root3Child0Tail1
    (if 2 ≤ root3Child0Candidates0.length then search edge 2 [13, 19, 577] root3Child0Candidates0 609 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root3Child0Scan1
theorem root3Child0 : search edge 3 [13, 19] root3Candidates0 32 seed = seed := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 2) 3 [13, 19] 32 root3Child0Tail0 seed = seed
  exact root3Child0Scan0
theorem root3Step0 : scan edge (search edge 3) 4 [13] 13 root3Tail0 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail1 seed := by
  rw [root3Tail0, scan, if_neg (by decide), root3Candidates0_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail1
    (if 3 ≤ root3Candidates0.length then search edge 3 [13, 19] root3Candidates0 32 seed else seed) = _
  rw [if_pos (by decide), root3Child0]
def root3Candidates1 : List Nat := [331, 2383, 2971, 5431, 5623, 8581, 19543, 20533]
set_option maxRecDepth 8192 in
theorem root3Candidates1_checked : candidates edge 4 13 61 root3Tail2 seed.bound = root3Candidates1 := by
  unfold candidates
  have row : edge 61 = (SortedNatTree.contains FastRows.row61) := by
    change Data.row61.contains = (SortedNatTree.contains FastRows.row61)
    exact FastRows.row61_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child1 : search edge 3 [13, 61] root3Candidates1 74 seed = seed := by decide
theorem root3Step1 : scan edge (search edge 3) 4 [13] 13 root3Tail1 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail2 seed := by
  rw [root3Tail1, scan, if_neg (by decide), root3Candidates1_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail2
    (if 3 ≤ root3Candidates1.length then search edge 3 [13, 61] root3Candidates1 74 seed else seed) = _
  rw [if_pos (by decide), root3Child1]
def root3Candidates2 : List Nat := [997, 8731, 10039, 17401, 17443, 22003]
set_option maxRecDepth 8192 in
theorem root3Candidates2_checked : candidates edge 4 13 103 root3Tail3 seed.bound = root3Candidates2 := by
  unfold candidates
  have row : edge 103 = (SortedNatTree.contains FastRows.row103) := by
    change Data.row103.contains = (SortedNatTree.contains FastRows.row103)
    exact FastRows.row103_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child2 : search edge 3 [13, 103] root3Candidates2 116 seed = seed := by decide
theorem root3Step2 : scan edge (search edge 3) 4 [13] 13 root3Tail2 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail3 seed := by
  rw [root3Tail2, scan, if_neg (by decide), root3Candidates2_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail3
    (if 3 ≤ root3Candidates2.length then search edge 3 [13, 103] root3Candidates2 116 seed else seed) = _
  rw [if_pos (by decide), root3Child2]
def root3Candidates3 : List Nat := [241, 331, 1129, 3673, 4363, 6733, 7723, 7993, 9547, 11287]
set_option maxRecDepth 8192 in
theorem root3Candidates3_checked : candidates edge 4 13 127 root3Tail4 seed.bound = root3Candidates3 := by
  unfold candidates
  have row : edge 127 = (SortedNatTree.contains FastRows.row127) := by
    change Data.row127.contains = (SortedNatTree.contains FastRows.row127)
    exact FastRows.row127_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child3 : search edge 3 [13, 127] root3Candidates3 140 seed = seed := by decide
theorem root3Step3 : scan edge (search edge 3) 4 [13] 13 root3Tail3 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail4 seed := by
  rw [root3Tail3, scan, if_neg (by decide), root3Candidates3_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail4
    (if 3 ≤ root3Candidates3.length then search edge 3 [13, 127] root3Candidates3 140 seed else seed) = _
  rw [if_pos (by decide), root3Child3]
def root3Candidates4 : List Nat := [5431, 9721, 9883, 17683]
set_option maxRecDepth 8192 in
theorem root3Candidates4_checked : candidates edge 4 13 241 root3Tail5 seed.bound = root3Candidates4 := by
  unfold candidates
  have row : edge 241 = (SortedNatTree.contains FastRows.row241) := by
    change Data.row241.contains = (SortedNatTree.contains FastRows.row241)
    exact FastRows.row241_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child4 : search edge 3 [13, 241] root3Candidates4 254 seed = seed := by decide
theorem root3Step4 : scan edge (search edge 3) 4 [13] 13 root3Tail4 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail5 seed := by
  rw [root3Tail4, scan, if_neg (by decide), root3Candidates4_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail5
    (if 3 ≤ root3Candidates4.length then search edge 3 [13, 241] root3Candidates4 254 seed else seed) = _
  rw [if_pos (by decide), root3Child4]
def root3Candidates5 : List Nat := [577, 4513, 10993, 17443, 19543, 20749, 24907]
set_option maxRecDepth 8192 in
theorem root3Candidates5_checked : candidates edge 4 13 331 root3Tail6 seed.bound = root3Candidates5 := by
  unfold candidates
  have row : edge 331 = (SortedNatTree.contains FastRows.row331) := by
    change Data.row331.contains = (SortedNatTree.contains FastRows.row331)
    exact FastRows.row331_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child5 : search edge 3 [13, 331] root3Candidates5 344 seed = seed := by decide
theorem root3Step5 : scan edge (search edge 3) 4 [13] 13 root3Tail5 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail6 seed := by
  rw [root3Tail5, scan, if_neg (by decide), root3Candidates5_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail6
    (if 3 ≤ root3Candidates5.length then search edge 3 [13, 331] root3Candidates5 344 seed else seed) = _
  rw [if_pos (by decide), root3Child5]
def root3Candidates6 : List Nat := [3499, 5431, 7723, 21139]
set_option maxRecDepth 8192 in
theorem root3Candidates6_checked : candidates edge 4 13 337 root3Tail7 seed.bound = root3Candidates6 := by
  unfold candidates
  have row : edge 337 = (SortedNatTree.contains FastRows.row337) := by
    change Data.row337.contains = (SortedNatTree.contains FastRows.row337)
    exact FastRows.row337_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child6 : search edge 3 [13, 337] root3Candidates6 350 seed = seed := by decide
theorem root3Step6 : scan edge (search edge 3) 4 [13] 13 root3Tail6 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail7 seed := by
  rw [root3Tail6, scan, if_neg (by decide), root3Candidates6_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail7
    (if 3 ≤ root3Candidates6.length then search edge 3 [13, 337] root3Candidates6 350 seed else seed) = _
  rw [if_pos (by decide), root3Child6]
def root3Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root3Candidates7_checked : candidates edge 4 13 367 root3Tail8 seed.bound = root3Candidates7 := by
  unfold candidates
  have row : edge 367 = (SortedNatTree.contains FastRows.row367) := by
    change Data.row367.contains = (SortedNatTree.contains FastRows.row367)
    exact FastRows.row367_checked.symm
  rw [row]
  decide
theorem root3Step7 : scan edge (search edge 3) 4 [13] 13 root3Tail7 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail8 seed := by
  rw [root3Tail7, scan, if_neg (by decide), root3Candidates7_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail8
    (if 3 ≤ root3Candidates7.length then search edge 3 [13, 367] root3Candidates7 380 seed else seed) = _
  rw [if_neg (by decide)]
def root3Candidates8 : List Nat := [577, 1129, 2113, 3499, 4591, 4909, 5851, 8581, 10891, 23893]
set_option maxRecDepth 8192 in
theorem root3Candidates8_checked : candidates edge 4 13 523 root3Tail9 seed.bound = root3Candidates8 := by
  unfold candidates
  have row : edge 523 = (SortedNatTree.contains FastRows.row523) := by
    change Data.row523.contains = (SortedNatTree.contains FastRows.row523)
    exact FastRows.row523_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child8 : search edge 3 [13, 523] root3Candidates8 536 seed = seed := by decide
theorem root3Step8 : scan edge (search edge 3) 4 [13] 13 root3Tail8 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail9 seed := by
  rw [root3Tail8, scan, if_neg (by decide), root3Candidates8_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail9
    (if 3 ≤ root3Candidates8.length then search edge 3 [13, 523] root3Candidates8 536 seed else seed) = _
  rw [if_pos (by decide), root3Child8]
def root3Candidates9 : List Nat := [5431, 10039]
set_option maxRecDepth 8192 in
theorem root3Candidates9_checked : candidates edge 4 13 577 root3Tail10 seed.bound = root3Candidates9 := by
  unfold candidates
  have row : edge 577 = (SortedNatTree.contains FastRows.row577) := by
    change Data.row577.contains = (SortedNatTree.contains FastRows.row577)
    exact FastRows.row577_checked.symm
  rw [row]
  decide
theorem root3Step9 : scan edge (search edge 3) 4 [13] 13 root3Tail9 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail10 seed := by
  rw [root3Tail9, scan, if_neg (by decide), root3Candidates9_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail10
    (if 3 ≤ root3Candidates9.length then search edge 3 [13, 577] root3Candidates9 590 seed else seed) = _
  rw [if_neg (by decide)]
def root3Candidates10 : List Nat := [1543, 10891, 20173]
set_option maxRecDepth 8192 in
theorem root3Candidates10_checked : candidates edge 4 13 709 root3Tail11 seed.bound = root3Candidates10 := by
  unfold candidates
  have row : edge 709 = (SortedNatTree.contains FastRows.row709) := by
    change Data.row709.contains = (SortedNatTree.contains FastRows.row709)
    exact FastRows.row709_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child10 : search edge 3 [13, 709] root3Candidates10 722 seed = seed := by decide
theorem root3Step10 : scan edge (search edge 3) 4 [13] 13 root3Tail10 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail11 seed := by
  rw [root3Tail10, scan, if_neg (by decide), root3Candidates10_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail11
    (if 3 ≤ root3Candidates10.length then search edge 3 [13, 709] root3Candidates10 722 seed else seed) = _
  rw [if_pos (by decide), root3Child10]
def root3Candidates11 : List Nat := [9091, 12517, 17683, 17929, 21961]
set_option maxRecDepth 8192 in
theorem root3Candidates11_checked : candidates edge 4 13 829 root3Tail12 seed.bound = root3Candidates11 := by
  unfold candidates
  have row : edge 829 = (SortedNatTree.contains FastRows.row829) := by
    change Data.row829.contains = (SortedNatTree.contains FastRows.row829)
    exact FastRows.row829_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child11 : search edge 3 [13, 829] root3Candidates11 842 seed = seed := by decide
theorem root3Step11 : scan edge (search edge 3) 4 [13] 13 root3Tail11 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail12 seed := by
  rw [root3Tail11, scan, if_neg (by decide), root3Candidates11_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail12
    (if 3 ≤ root3Candidates11.length then search edge 3 [13, 829] root3Candidates11 842 seed else seed) = _
  rw [if_pos (by decide), root3Child11]
def root3Candidates12 : List Nat := [4243, 4513, 9091, 20287, 22171]
set_option maxRecDepth 8192 in
theorem root3Candidates12_checked : candidates edge 4 13 997 root3Tail13 seed.bound = root3Candidates12 := by
  unfold candidates
  have row : edge 997 = (SortedNatTree.contains FastRows.row997) := by
    change Data.row997.contains = (SortedNatTree.contains FastRows.row997)
    exact FastRows.row997_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child12 : search edge 3 [13, 997] root3Candidates12 1010 seed = seed := by decide
theorem root3Step12 : scan edge (search edge 3) 4 [13] 13 root3Tail12 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail13 seed := by
  rw [root3Tail12, scan, if_neg (by decide), root3Candidates12_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail13
    (if 3 ≤ root3Candidates12.length then search edge 3 [13, 997] root3Candidates12 1010 seed else seed) = _
  rw [if_pos (by decide), root3Child12]
def root3Candidates13 : List Nat := [2851, 6043, 6397, 16831]
set_option maxRecDepth 8192 in
theorem root3Candidates13_checked : candidates edge 4 13 1009 root3Tail14 seed.bound = root3Candidates13 := by
  unfold candidates
  have row : edge 1009 = (SortedNatTree.contains FastRows.row1009) := by
    change Data.row1009.contains = (SortedNatTree.contains FastRows.row1009)
    exact FastRows.row1009_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child13 : search edge 3 [13, 1009] root3Candidates13 1022 seed = seed := by decide
theorem root3Step13 : scan edge (search edge 3) 4 [13] 13 root3Tail13 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail14 seed := by
  rw [root3Tail13, scan, if_neg (by decide), root3Candidates13_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail14
    (if 3 ≤ root3Candidates13.length then search edge 3 [13, 1009] root3Candidates13 1022 seed else seed) = _
  rw [if_pos (by decide), root3Child13]
def root3Candidates14 : List Nat := [6421, 8629, 8731, 9697]
set_option maxRecDepth 8192 in
theorem root3Candidates14_checked : candidates edge 4 13 1129 root3Tail15 seed.bound = root3Candidates14 := by
  unfold candidates
  have row : edge 1129 = (SortedNatTree.contains FastRows.row1129) := by
    change Data.row1129.contains = (SortedNatTree.contains FastRows.row1129)
    exact FastRows.row1129_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child14 : search edge 3 [13, 1129] root3Candidates14 1142 seed = seed := by decide
theorem root3Step14 : scan edge (search edge 3) 4 [13] 13 root3Tail14 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail15 seed := by
  rw [root3Tail14, scan, if_neg (by decide), root3Candidates14_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail15
    (if 3 ≤ root3Candidates14.length then search edge 3 [13, 1129] root3Candidates14 1142 seed else seed) = _
  rw [if_pos (by decide), root3Child14]
def root3Candidates15 : List Nat := [2851, 6043, 13999, 21379]
set_option maxRecDepth 8192 in
theorem root3Candidates15_checked : candidates edge 4 13 1213 root3Tail16 seed.bound = root3Candidates15 := by
  unfold candidates
  have row : edge 1213 = (SortedNatTree.contains FastRows.row1213) := by
    change Data.row1213.contains = (SortedNatTree.contains FastRows.row1213)
    exact FastRows.row1213_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child15 : search edge 3 [13, 1213] root3Candidates15 1226 seed = seed := by decide
theorem root3Step15 : scan edge (search edge 3) 4 [13] 13 root3Tail15 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail16 seed := by
  rw [root3Tail15, scan, if_neg (by decide), root3Candidates15_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail16
    (if 3 ≤ root3Candidates15.length then search edge 3 [13, 1213] root3Candidates15 1226 seed else seed) = _
  rw [if_pos (by decide), root3Child15]
def root3Candidates16 : List Nat := [2689, 4243, 4327, 4789, 5647, 13567, 13999]
set_option maxRecDepth 8192 in
theorem root3Candidates16_checked : candidates edge 4 13 1231 root3Tail17 seed.bound = root3Candidates16 := by
  unfold candidates
  have row : edge 1231 = (SortedNatTree.contains FastRows.row1231) := by
    change Data.row1231.contains = (SortedNatTree.contains FastRows.row1231)
    exact FastRows.row1231_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child16 : search edge 3 [13, 1231] root3Candidates16 1244 seed = seed := by decide
theorem root3Step16 : scan edge (search edge 3) 4 [13] 13 root3Tail16 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail17 seed := by
  rw [root3Tail16, scan, if_neg (by decide), root3Candidates16_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail17
    (if 3 ≤ root3Candidates16.length then search edge 3 [13, 1231] root3Candidates16 1244 seed else seed) = _
  rw [if_pos (by decide), root3Child16]
def root3Candidates17 : List Nat := [1381, 5581, 12601, 14779, 16417]
set_option maxRecDepth 8192 in
theorem root3Candidates17_checked : candidates edge 4 13 1321 root3Tail18 seed.bound = root3Candidates17 := by
  unfold candidates
  have row : edge 1321 = (SortedNatTree.contains FastRows.row1321) := by
    change Data.row1321.contains = (SortedNatTree.contains FastRows.row1321)
    exact FastRows.row1321_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child17 : search edge 3 [13, 1321] root3Candidates17 1334 seed = seed := by decide
theorem root3Step17 : scan edge (search edge 3) 4 [13] 13 root3Tail17 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail18 seed := by
  rw [root3Tail17, scan, if_neg (by decide), root3Candidates17_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail18
    (if 3 ≤ root3Candidates17.length then search edge 3 [13, 1321] root3Candidates17 1334 seed else seed) = _
  rw [if_pos (by decide), root3Child17]
def root3Candidates18 : List Nat := [9199]
set_option maxRecDepth 8192 in
theorem root3Candidates18_checked : candidates edge 4 13 1381 root3Tail19 seed.bound = root3Candidates18 := by
  unfold candidates
  have row : edge 1381 = (SortedNatTree.contains FastRows.row1381) := by
    change Data.row1381.contains = (SortedNatTree.contains FastRows.row1381)
    exact FastRows.row1381_checked.symm
  rw [row]
  decide
theorem root3Step18 : scan edge (search edge 3) 4 [13] 13 root3Tail18 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail19 seed := by
  rw [root3Tail18, scan, if_neg (by decide), root3Candidates18_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail19
    (if 3 ≤ root3Candidates18.length then search edge 3 [13, 1381] root3Candidates18 1394 seed else seed) = _
  rw [if_neg (by decide)]
def root3Candidates19 : List Nat := [8629, 9397, 9697, 12601, 14149, 20749, 21379]
set_option maxRecDepth 8192 in
theorem root3Candidates19_checked : candidates edge 4 13 1489 root3Tail20 seed.bound = root3Candidates19 := by
  unfold candidates
  have row : edge 1489 = (SortedNatTree.contains FastRows.row1489) := by
    change Data.row1489.contains = (SortedNatTree.contains FastRows.row1489)
    exact FastRows.row1489_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child19 : search edge 3 [13, 1489] root3Candidates19 1502 seed = seed := by decide
theorem root3Step19 : scan edge (search edge 3) 4 [13] 13 root3Tail19 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail20 seed := by
  rw [root3Tail19, scan, if_neg (by decide), root3Candidates19_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail20
    (if 3 ≤ root3Candidates19.length then search edge 3 [13, 1489] root3Candidates19 1502 seed else seed) = _
  rw [if_pos (by decide), root3Child19]
def root3Candidates20 : List Nat := [3967, 8053, 9883, 10369, 16417, 19609, 22171]
set_option maxRecDepth 8192 in
theorem root3Candidates20_checked : candidates edge 4 13 1543 root3Tail21 seed.bound = root3Candidates20 := by
  unfold candidates
  have row : edge 1543 = (SortedNatTree.contains FastRows.row1543) := by
    change Data.row1543.contains = (SortedNatTree.contains FastRows.row1543)
    exact FastRows.row1543_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child20 : search edge 3 [13, 1543] root3Candidates20 1556 seed = seed := by decide
theorem root3Step20 : scan edge (search edge 3) 4 [13] 13 root3Tail20 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail21 seed := by
  rw [root3Tail20, scan, if_neg (by decide), root3Candidates20_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail21
    (if 3 ≤ root3Candidates20.length then search edge 3 [13, 1543] root3Candidates20 1556 seed else seed) = _
  rw [if_pos (by decide), root3Child20]
def root3Candidates21 : List Nat := [4591, 7507, 8461, 14479, 18301]
set_option maxRecDepth 8192 in
theorem root3Candidates21_checked : candidates edge 4 13 1627 root3Tail22 seed.bound = root3Candidates21 := by
  unfold candidates
  have row : edge 1627 = (SortedNatTree.contains FastRows.row1627) := by
    change Data.row1627.contains = (SortedNatTree.contains FastRows.row1627)
    exact FastRows.row1627_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child21 : search edge 3 [13, 1627] root3Candidates21 1640 seed = seed := by decide
theorem root3Step21 : scan edge (search edge 3) 4 [13] 13 root3Tail21 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail22 seed := by
  rw [root3Tail21, scan, if_neg (by decide), root3Candidates21_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail22
    (if 3 ≤ root3Candidates21.length then search edge 3 [13, 1627] root3Candidates21 1640 seed else seed) = _
  rw [if_pos (by decide), root3Child21]
def root3Candidates22 : List Nat := [2383, 4789, 12211]
set_option maxRecDepth 8192 in
theorem root3Candidates22_checked : candidates edge 4 13 1861 root3Tail23 seed.bound = root3Candidates22 := by
  unfold candidates
  have row : edge 1861 = (SortedNatTree.contains FastRows.row1861) := by
    change Data.row1861.contains = (SortedNatTree.contains FastRows.row1861)
    exact FastRows.row1861_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child22 : search edge 3 [13, 1861] root3Candidates22 1874 seed = seed := by decide
theorem root3Step22 : scan edge (search edge 3) 4 [13] 13 root3Tail22 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail23 seed := by
  rw [root3Tail22, scan, if_neg (by decide), root3Candidates22_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail23
    (if 3 ≤ root3Candidates22.length then search edge 3 [13, 1861] root3Candidates22 1874 seed else seed) = _
  rw [if_pos (by decide), root3Child22]
def root3Candidates23 : List Nat := [3583, 5431, 7927, 9439, 21139, 21157, 21379]
set_option maxRecDepth 8192 in
theorem root3Candidates23_checked : candidates edge 4 13 2113 root3Tail24 seed.bound = root3Candidates23 := by
  unfold candidates
  have row : edge 2113 = (SortedNatTree.contains FastRows.row2113) := by
    change Data.row2113.contains = (SortedNatTree.contains FastRows.row2113)
    exact FastRows.row2113_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child23 : search edge 3 [13, 2113] root3Candidates23 2126 seed = seed := by decide
theorem root3Step23 : scan edge (search edge 3) 4 [13] 13 root3Tail23 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail24 seed := by
  rw [root3Tail23, scan, if_neg (by decide), root3Candidates23_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail24
    (if 3 ≤ root3Candidates23.length then search edge 3 [13, 2113] root3Candidates23 2126 seed else seed) = _
  rw [if_pos (by decide), root3Child23]
def root3Candidates24 : List Nat := [4513, 6991, 17929]
set_option maxRecDepth 8192 in
theorem root3Candidates24_checked : candidates edge 4 13 2137 root3Tail25 seed.bound = root3Candidates24 := by
  unfold candidates
  have row : edge 2137 = (SortedNatTree.contains FastRows.row2137) := by
    change Data.row2137.contains = (SortedNatTree.contains FastRows.row2137)
    exact FastRows.row2137_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child24 : search edge 3 [13, 2137] root3Candidates24 2150 seed = seed := by decide
theorem root3Step24 : scan edge (search edge 3) 4 [13] 13 root3Tail24 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail25 seed := by
  rw [root3Tail24, scan, if_neg (by decide), root3Candidates24_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail25
    (if 3 ≤ root3Candidates24.length then search edge 3 [13, 2137] root3Candidates24 2150 seed else seed) = _
  rw [if_pos (by decide), root3Child24]
def root3Candidates25 : List Nat := [3187, 5077, 5431, 13999, 22147]
set_option maxRecDepth 8192 in
theorem root3Candidates25_checked : candidates edge 4 13 2287 root3Tail26 seed.bound = root3Candidates25 := by
  unfold candidates
  have row : edge 2287 = (SortedNatTree.contains FastRows.row2287) := by
    change Data.row2287.contains = (SortedNatTree.contains FastRows.row2287)
    exact FastRows.row2287_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child25 : search edge 3 [13, 2287] root3Candidates25 2300 seed = seed := by decide
theorem root3Step25 : scan edge (search edge 3) 4 [13] 13 root3Tail25 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail26 seed := by
  rw [root3Tail25, scan, if_neg (by decide), root3Candidates25_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail26
    (if 3 ≤ root3Candidates25.length then search edge 3 [13, 2287] root3Candidates25 2300 seed else seed) = _
  rw [if_pos (by decide), root3Child25]
def root3Candidates26 : List Nat := [2851, 4327, 11311, 12277, 20749, 22003]
set_option maxRecDepth 8192 in
theorem root3Candidates26_checked : candidates edge 4 13 2347 root3Tail27 seed.bound = root3Candidates26 := by
  unfold candidates
  have row : edge 2347 = (SortedNatTree.contains FastRows.row2347) := by
    change Data.row2347.contains = (SortedNatTree.contains FastRows.row2347)
    exact FastRows.row2347_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child26 : search edge 3 [13, 2347] root3Candidates26 2360 seed = seed := by decide
theorem root3Step26 : scan edge (search edge 3) 4 [13] 13 root3Tail26 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail27 seed := by
  rw [root3Tail26, scan, if_neg (by decide), root3Candidates26_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail27
    (if 3 ≤ root3Candidates26.length then search edge 3 [13, 2347] root3Candidates26 2360 seed else seed) = _
  rw [if_pos (by decide), root3Child26]
def root3Candidates27 : List Nat := [3583, 5431, 6733, 8599, 14437]
set_option maxRecDepth 8192 in
theorem root3Candidates27_checked : candidates edge 4 13 2383 root3Tail28 seed.bound = root3Candidates27 := by
  unfold candidates
  have row : edge 2383 = (SortedNatTree.contains FastRows.row2383) := by
    change Data.row2383.contains = (SortedNatTree.contains FastRows.row2383)
    exact FastRows.row2383_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child27 : search edge 3 [13, 2383] root3Candidates27 2396 seed = seed := by decide
theorem root3Step27 : scan edge (search edge 3) 4 [13] 13 root3Tail27 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail28 seed := by
  rw [root3Tail27, scan, if_neg (by decide), root3Candidates27_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail28
    (if 3 ≤ root3Candidates27.length then search edge 3 [13, 2383] root3Candidates27 2396 seed else seed) = _
  rw [if_pos (by decide), root3Child27]
def root3Candidates28 : List Nat := [6043, 7477, 10891, 17989]
set_option maxRecDepth 8192 in
theorem root3Candidates28_checked : candidates edge 4 13 2689 root3Tail29 seed.bound = root3Candidates28 := by
  unfold candidates
  have row : edge 2689 = (SortedNatTree.contains FastRows.row2689) := by
    change Data.row2689.contains = (SortedNatTree.contains FastRows.row2689)
    exact FastRows.row2689_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child28 : search edge 3 [13, 2689] root3Candidates28 2702 seed = seed := by decide
theorem root3Step28 : scan edge (search edge 3) 4 [13] 13 root3Tail28 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail29 seed := by
  rw [root3Tail28, scan, if_neg (by decide), root3Candidates28_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail29
    (if 3 ≤ root3Candidates28.length then search edge 3 [13, 2689] root3Candidates28 2702 seed else seed) = _
  rw [if_pos (by decide), root3Child28]
def root3Candidates29 : List Nat := [3187, 4951, 9697, 19609]
set_option maxRecDepth 8192 in
theorem root3Candidates29_checked : candidates edge 4 13 2851 root3Tail30 seed.bound = root3Candidates29 := by
  unfold candidates
  have row : edge 2851 = (SortedNatTree.contains FastRows.row2851) := by
    change Data.row2851.contains = (SortedNatTree.contains FastRows.row2851)
    exact FastRows.row2851_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child29 : search edge 3 [13, 2851] root3Candidates29 2864 seed = seed := by decide
theorem root3Step29 : scan edge (search edge 3) 4 [13] 13 root3Tail29 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail30 seed := by
  rw [root3Tail29, scan, if_neg (by decide), root3Candidates29_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail30
    (if 3 ≤ root3Candidates29.length then search edge 3 [13, 2851] root3Candidates29 2864 seed else seed) = _
  rw [if_pos (by decide), root3Child29]
def root3Candidates30 : List Nat := [6733, 8461, 12517, 18553]
set_option maxRecDepth 8192 in
theorem root3Candidates30_checked : candidates edge 4 13 2953 root3Tail31 seed.bound = root3Candidates30 := by
  unfold candidates
  have row : edge 2953 = (SortedNatTree.contains FastRows.row2953) := by
    change Data.row2953.contains = (SortedNatTree.contains FastRows.row2953)
    exact FastRows.row2953_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root3Child30 : search edge 3 [13, 2953] root3Candidates30 2966 seed = seed := by decide
theorem root3Step30 : scan edge (search edge 3) 4 [13] 13 root3Tail30 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail31 seed := by
  rw [root3Tail30, scan, if_neg (by decide), root3Candidates30_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail31
    (if 3 ≤ root3Candidates30.length then search edge 3 [13, 2953] root3Candidates30 2966 seed else seed) = _
  rw [if_pos (by decide), root3Child30]
def root3Candidates31 : List Nat := [17419]
set_option maxRecDepth 8192 in
theorem root3Candidates31_checked : candidates edge 4 13 2971 root3Tail32 seed.bound = root3Candidates31 := by
  unfold candidates
  have row : edge 2971 = (SortedNatTree.contains FastRows.row2971) := by
    change Data.row2971.contains = (SortedNatTree.contains FastRows.row2971)
    exact FastRows.row2971_checked.symm
  rw [row]
  decide
theorem root3Step31 : scan edge (search edge 3) 4 [13] 13 root3Tail31 seed =
    scan edge (search edge 3) 4 [13] 13 root3Tail32 seed := by
  rw [root3Tail31, scan, if_neg (by decide), root3Candidates31_checked]
  change scan edge (search edge 3) 4 [13] 13 root3Tail32
    (if 3 ≤ root3Candidates31.length then search edge 3 [13, 2971] root3Candidates31 2984 seed else seed) = _
  rw [if_neg (by decide)]
end ProjectEulerSolutions.PrimePairLarge.Search
