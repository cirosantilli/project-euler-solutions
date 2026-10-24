import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates456 : List Nat := [3517, 3547, 3607, 4219, 4231, 4339, 4357, 5197, 5479, 5557, 5851, 6043, 6067, 6571, 7717, 7873, 7927, 8263]
set_option maxRecDepth 4096 in
theorem rootCandidates456_checked : candidates edge 5 0 3253 tail457 best.bound = rootCandidates456 := by
  unfold candidates
  have row : edge 3253 = SearchData.row3253.contains := by funext q; rfl
  rw [row]
  decide

def innerTail456_18 : List Nat := []
theorem innerScan456_18 : scan edge (search edge 3) 4 [3253] 3253 innerTail456_18 best = best := rfl

def innerTail456_17 : List Nat := 8263 :: innerTail456_18
def innerCandidates456_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates456_17_checked : candidates edge 4 3253 8263 innerTail456_18 best.bound = innerCandidates456_17 := by
  unfold candidates
  have row : edge 8263 = SearchData.row8263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan456_17 : scan edge (search edge 3) 4 [3253] 3253 innerTail456_17 best = best := by
  rw [innerTail456_17, scan, if_neg (by decide), innerCandidates456_17_checked]
  change scan edge (search edge 3) 4 [3253] 3253 innerTail456_18
    (if 3 ≤ innerCandidates456_17.length then search edge 3 [3253, 8263] innerCandidates456_17 11516 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan456_18

def innerTail456_16 : List Nat := 7927 :: innerTail456_17
def innerCandidates456_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates456_16_checked : candidates edge 4 3253 7927 innerTail456_17 best.bound = innerCandidates456_16 := by
  unfold candidates
  have row : edge 7927 = SearchData.row7927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan456_16 : scan edge (search edge 3) 4 [3253] 3253 innerTail456_16 best = best := by
  rw [innerTail456_16, scan, if_neg (by decide), innerCandidates456_16_checked]
  change scan edge (search edge 3) 4 [3253] 3253 innerTail456_17
    (if 3 ≤ innerCandidates456_16.length then search edge 3 [3253, 7927] innerCandidates456_16 11180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan456_17

def innerTail456_15 : List Nat := 7873 :: innerTail456_16
def innerCandidates456_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates456_15_checked : candidates edge 4 3253 7873 innerTail456_16 best.bound = innerCandidates456_15 := by
  unfold candidates
  have row : edge 7873 = SearchData.row7873.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan456_15 : scan edge (search edge 3) 4 [3253] 3253 innerTail456_15 best = best := by
  rw [innerTail456_15, scan, if_neg (by decide), innerCandidates456_15_checked]
  change scan edge (search edge 3) 4 [3253] 3253 innerTail456_16
    (if 3 ≤ innerCandidates456_15.length then search edge 3 [3253, 7873] innerCandidates456_15 11126 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan456_16

def innerTail456_14 : List Nat := 7717 :: innerTail456_15
def innerCandidates456_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates456_14_checked : candidates edge 4 3253 7717 innerTail456_15 best.bound = innerCandidates456_14 := by
  unfold candidates
  have row : edge 7717 = SearchData.row7717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan456_14 : scan edge (search edge 3) 4 [3253] 3253 innerTail456_14 best = best := by
  rw [innerTail456_14, scan, if_neg (by decide), innerCandidates456_14_checked]
  change scan edge (search edge 3) 4 [3253] 3253 innerTail456_15
    (if 3 ≤ innerCandidates456_14.length then search edge 3 [3253, 7717] innerCandidates456_14 10970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan456_15

def innerTail456_13 : List Nat := 6571 :: innerTail456_14
def innerCandidates456_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates456_13_checked : candidates edge 4 3253 6571 innerTail456_14 best.bound = innerCandidates456_13 := by
  unfold candidates
  have row : edge 6571 = SearchData.row6571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan456_13 : scan edge (search edge 3) 4 [3253] 3253 innerTail456_13 best = best := by
  rw [innerTail456_13, scan, if_neg (by decide), innerCandidates456_13_checked]
  change scan edge (search edge 3) 4 [3253] 3253 innerTail456_14
    (if 3 ≤ innerCandidates456_13.length then search edge 3 [3253, 6571] innerCandidates456_13 9824 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan456_14

def innerTail456_12 : List Nat := 6067 :: innerTail456_13
def innerCandidates456_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates456_12_checked : candidates edge 4 3253 6067 innerTail456_13 best.bound = innerCandidates456_12 := by
  unfold candidates
  have row : edge 6067 = SearchData.row6067.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan456_12 : scan edge (search edge 3) 4 [3253] 3253 innerTail456_12 best = best := by
  rw [innerTail456_12, scan, if_neg (by decide), innerCandidates456_12_checked]
  change scan edge (search edge 3) 4 [3253] 3253 innerTail456_13
    (if 3 ≤ innerCandidates456_12.length then search edge 3 [3253, 6067] innerCandidates456_12 9320 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan456_13

def innerTail456_11 : List Nat := 6043 :: innerTail456_12
def innerCandidates456_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates456_11_checked : candidates edge 4 3253 6043 innerTail456_12 best.bound = innerCandidates456_11 := by
  unfold candidates
  have row : edge 6043 = SearchData.row6043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan456_11 : scan edge (search edge 3) 4 [3253] 3253 innerTail456_11 best = best := by
  rw [innerTail456_11, scan, if_neg (by decide), innerCandidates456_11_checked]
  change scan edge (search edge 3) 4 [3253] 3253 innerTail456_12
    (if 3 ≤ innerCandidates456_11.length then search edge 3 [3253, 6043] innerCandidates456_11 9296 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan456_12

def innerTail456_10 : List Nat := 5851 :: innerTail456_11
def innerCandidates456_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates456_10_checked : candidates edge 4 3253 5851 innerTail456_11 best.bound = innerCandidates456_10 := by
  unfold candidates
  have row : edge 5851 = SearchData.row5851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan456_10 : scan edge (search edge 3) 4 [3253] 3253 innerTail456_10 best = best := by
  rw [innerTail456_10, scan, if_neg (by decide), innerCandidates456_10_checked]
  change scan edge (search edge 3) 4 [3253] 3253 innerTail456_11
    (if 3 ≤ innerCandidates456_10.length then search edge 3 [3253, 5851] innerCandidates456_10 9104 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan456_11

def innerTail456_9 : List Nat := 5557 :: innerTail456_10
def innerCandidates456_9 : List Nat := [6067]
set_option maxRecDepth 4096 in
theorem innerCandidates456_9_checked : candidates edge 4 3253 5557 innerTail456_10 best.bound = innerCandidates456_9 := by
  unfold candidates
  have row : edge 5557 = SearchData.row5557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan456_9 : scan edge (search edge 3) 4 [3253] 3253 innerTail456_9 best = best := by
  rw [innerTail456_9, scan, if_neg (by decide), innerCandidates456_9_checked]
  change scan edge (search edge 3) 4 [3253] 3253 innerTail456_10
    (if 3 ≤ innerCandidates456_9.length then search edge 3 [3253, 5557] innerCandidates456_9 8810 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan456_10

def innerTail456_8 : List Nat := 5479 :: innerTail456_9
def innerCandidates456_8 : List Nat := [7927]
set_option maxRecDepth 4096 in
theorem innerCandidates456_8_checked : candidates edge 4 3253 5479 innerTail456_9 best.bound = innerCandidates456_8 := by
  unfold candidates
  have row : edge 5479 = SearchData.row5479.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan456_8 : scan edge (search edge 3) 4 [3253] 3253 innerTail456_8 best = best := by
  rw [innerTail456_8, scan, if_neg (by decide), innerCandidates456_8_checked]
  change scan edge (search edge 3) 4 [3253] 3253 innerTail456_9
    (if 3 ≤ innerCandidates456_8.length then search edge 3 [3253, 5479] innerCandidates456_8 8732 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan456_9

def innerTail456_7 : List Nat := 5197 :: innerTail456_8
def innerCandidates456_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates456_7_checked : candidates edge 4 3253 5197 innerTail456_8 best.bound = innerCandidates456_7 := by
  unfold candidates
  have row : edge 5197 = SearchData.row5197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan456_7 : scan edge (search edge 3) 4 [3253] 3253 innerTail456_7 best = best := by
  rw [innerTail456_7, scan, if_neg (by decide), innerCandidates456_7_checked]
  change scan edge (search edge 3) 4 [3253] 3253 innerTail456_8
    (if 3 ≤ innerCandidates456_7.length then search edge 3 [3253, 5197] innerCandidates456_7 8450 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan456_8

def innerTail456_6 : List Nat := 4357 :: innerTail456_7
def innerCandidates456_6 : List Nat := [5479]
set_option maxRecDepth 4096 in
theorem innerCandidates456_6_checked : candidates edge 4 3253 4357 innerTail456_7 best.bound = innerCandidates456_6 := by
  unfold candidates
  have row : edge 4357 = SearchData.row4357.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan456_6 : scan edge (search edge 3) 4 [3253] 3253 innerTail456_6 best = best := by
  rw [innerTail456_6, scan, if_neg (by decide), innerCandidates456_6_checked]
  change scan edge (search edge 3) 4 [3253] 3253 innerTail456_7
    (if 3 ≤ innerCandidates456_6.length then search edge 3 [3253, 4357] innerCandidates456_6 7610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan456_7

def innerTail456_5 : List Nat := 4339 :: innerTail456_6
def innerCandidates456_5 : List Nat := [7927]
set_option maxRecDepth 4096 in
theorem innerCandidates456_5_checked : candidates edge 4 3253 4339 innerTail456_6 best.bound = innerCandidates456_5 := by
  unfold candidates
  have row : edge 4339 = SearchData.row4339.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan456_5 : scan edge (search edge 3) 4 [3253] 3253 innerTail456_5 best = best := by
  rw [innerTail456_5, scan, if_neg (by decide), innerCandidates456_5_checked]
  change scan edge (search edge 3) 4 [3253] 3253 innerTail456_6
    (if 3 ≤ innerCandidates456_5.length then search edge 3 [3253, 4339] innerCandidates456_5 7592 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan456_6

def innerTail456_4 : List Nat := 4231 :: innerTail456_5
def innerCandidates456_4 : List Nat := [4339, 5197]
set_option maxRecDepth 4096 in
theorem innerCandidates456_4_checked : candidates edge 4 3253 4231 innerTail456_5 best.bound = innerCandidates456_4 := by
  unfold candidates
  have row : edge 4231 = SearchData.row4231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan456_4 : scan edge (search edge 3) 4 [3253] 3253 innerTail456_4 best = best := by
  rw [innerTail456_4, scan, if_neg (by decide), innerCandidates456_4_checked]
  change scan edge (search edge 3) 4 [3253] 3253 innerTail456_5
    (if 3 ≤ innerCandidates456_4.length then search edge 3 [3253, 4231] innerCandidates456_4 7484 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan456_5

def innerTail456_3 : List Nat := 4219 :: innerTail456_4
def innerCandidates456_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates456_3_checked : candidates edge 4 3253 4219 innerTail456_4 best.bound = innerCandidates456_3 := by
  unfold candidates
  have row : edge 4219 = SearchData.row4219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan456_3 : scan edge (search edge 3) 4 [3253] 3253 innerTail456_3 best = best := by
  rw [innerTail456_3, scan, if_neg (by decide), innerCandidates456_3_checked]
  change scan edge (search edge 3) 4 [3253] 3253 innerTail456_4
    (if 3 ≤ innerCandidates456_3.length then search edge 3 [3253, 4219] innerCandidates456_3 7472 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan456_4

def innerTail456_2 : List Nat := 3607 :: innerTail456_3
def innerCandidates456_2 : List Nat := [6571]
set_option maxRecDepth 4096 in
theorem innerCandidates456_2_checked : candidates edge 4 3253 3607 innerTail456_3 best.bound = innerCandidates456_2 := by
  unfold candidates
  have row : edge 3607 = SearchData.row3607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan456_2 : scan edge (search edge 3) 4 [3253] 3253 innerTail456_2 best = best := by
  rw [innerTail456_2, scan, if_neg (by decide), innerCandidates456_2_checked]
  change scan edge (search edge 3) 4 [3253] 3253 innerTail456_3
    (if 3 ≤ innerCandidates456_2.length then search edge 3 [3253, 3607] innerCandidates456_2 6860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan456_3

def innerTail456_1 : List Nat := 3547 :: innerTail456_2
def innerCandidates456_1 : List Nat := [4357]
set_option maxRecDepth 4096 in
theorem innerCandidates456_1_checked : candidates edge 4 3253 3547 innerTail456_2 best.bound = innerCandidates456_1 := by
  unfold candidates
  have row : edge 3547 = SearchData.row3547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan456_1 : scan edge (search edge 3) 4 [3253] 3253 innerTail456_1 best = best := by
  rw [innerTail456_1, scan, if_neg (by decide), innerCandidates456_1_checked]
  change scan edge (search edge 3) 4 [3253] 3253 innerTail456_2
    (if 3 ≤ innerCandidates456_1.length then search edge 3 [3253, 3547] innerCandidates456_1 6800 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan456_2

def innerTail456_0 : List Nat := 3517 :: innerTail456_1
def innerCandidates456_0 : List Nat := [7717, 8263]
set_option maxRecDepth 4096 in
theorem innerCandidates456_0_checked : candidates edge 4 3253 3517 innerTail456_1 best.bound = innerCandidates456_0 := by
  unfold candidates
  have row : edge 3517 = SearchData.row3517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan456_0 : scan edge (search edge 3) 4 [3253] 3253 innerTail456_0 best = best := by
  rw [innerTail456_0, scan, if_neg (by decide), innerCandidates456_0_checked]
  change scan edge (search edge 3) 4 [3253] 3253 innerTail456_1
    (if 3 ≤ innerCandidates456_0.length then search edge 3 [3253, 3517] innerCandidates456_0 6770 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan456_1

theorem rootBranch456 : search edge 4 [3253] rootCandidates456 3253 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3253] 3253 innerTail456_0 best = best
  exact innerScan456_0

def rootCandidates455 : List Nat := [3413, 4241, 4253, 4409, 4721, 4973, 5039, 7229, 8273]
set_option maxRecDepth 4096 in
theorem rootCandidates455_checked : candidates edge 5 0 3251 tail456 best.bound = rootCandidates455 := by
  unfold candidates
  have row : edge 3251 = SearchData.row3251.contains := by funext q; rfl
  rw [row]
  decide

def innerTail455_9 : List Nat := []
theorem innerScan455_9 : scan edge (search edge 3) 4 [3251] 3251 innerTail455_9 best = best := rfl

def innerTail455_8 : List Nat := 8273 :: innerTail455_9
def innerCandidates455_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates455_8_checked : candidates edge 4 3251 8273 innerTail455_9 best.bound = innerCandidates455_8 := by
  unfold candidates
  have row : edge 8273 = SearchData.row8273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan455_8 : scan edge (search edge 3) 4 [3251] 3251 innerTail455_8 best = best := by
  rw [innerTail455_8, scan, if_neg (by decide), innerCandidates455_8_checked]
  change scan edge (search edge 3) 4 [3251] 3251 innerTail455_9
    (if 3 ≤ innerCandidates455_8.length then search edge 3 [3251, 8273] innerCandidates455_8 11524 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan455_9

def innerTail455_7 : List Nat := 7229 :: innerTail455_8
def innerCandidates455_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates455_7_checked : candidates edge 4 3251 7229 innerTail455_8 best.bound = innerCandidates455_7 := by
  unfold candidates
  have row : edge 7229 = SearchData.row7229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan455_7 : scan edge (search edge 3) 4 [3251] 3251 innerTail455_7 best = best := by
  rw [innerTail455_7, scan, if_neg (by decide), innerCandidates455_7_checked]
  change scan edge (search edge 3) 4 [3251] 3251 innerTail455_8
    (if 3 ≤ innerCandidates455_7.length then search edge 3 [3251, 7229] innerCandidates455_7 10480 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan455_8

def innerTail455_6 : List Nat := 5039 :: innerTail455_7
def innerCandidates455_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates455_6_checked : candidates edge 4 3251 5039 innerTail455_7 best.bound = innerCandidates455_6 := by
  unfold candidates
  have row : edge 5039 = SearchData.row5039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan455_6 : scan edge (search edge 3) 4 [3251] 3251 innerTail455_6 best = best := by
  rw [innerTail455_6, scan, if_neg (by decide), innerCandidates455_6_checked]
  change scan edge (search edge 3) 4 [3251] 3251 innerTail455_7
    (if 3 ≤ innerCandidates455_6.length then search edge 3 [3251, 5039] innerCandidates455_6 8290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan455_7

def innerTail455_5 : List Nat := 4973 :: innerTail455_6
def innerCandidates455_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates455_5_checked : candidates edge 4 3251 4973 innerTail455_6 best.bound = innerCandidates455_5 := by
  unfold candidates
  have row : edge 4973 = SearchData.row4973.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan455_5 : scan edge (search edge 3) 4 [3251] 3251 innerTail455_5 best = best := by
  rw [innerTail455_5, scan, if_neg (by decide), innerCandidates455_5_checked]
  change scan edge (search edge 3) 4 [3251] 3251 innerTail455_6
    (if 3 ≤ innerCandidates455_5.length then search edge 3 [3251, 4973] innerCandidates455_5 8224 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan455_6

def innerTail455_4 : List Nat := 4721 :: innerTail455_5
def innerCandidates455_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates455_4_checked : candidates edge 4 3251 4721 innerTail455_5 best.bound = innerCandidates455_4 := by
  unfold candidates
  have row : edge 4721 = SearchData.row4721.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan455_4 : scan edge (search edge 3) 4 [3251] 3251 innerTail455_4 best = best := by
  rw [innerTail455_4, scan, if_neg (by decide), innerCandidates455_4_checked]
  change scan edge (search edge 3) 4 [3251] 3251 innerTail455_5
    (if 3 ≤ innerCandidates455_4.length then search edge 3 [3251, 4721] innerCandidates455_4 7972 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan455_5

def innerTail455_3 : List Nat := 4409 :: innerTail455_4
def innerCandidates455_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates455_3_checked : candidates edge 4 3251 4409 innerTail455_4 best.bound = innerCandidates455_3 := by
  unfold candidates
  have row : edge 4409 = SearchData.row4409.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan455_3 : scan edge (search edge 3) 4 [3251] 3251 innerTail455_3 best = best := by
  rw [innerTail455_3, scan, if_neg (by decide), innerCandidates455_3_checked]
  change scan edge (search edge 3) 4 [3251] 3251 innerTail455_4
    (if 3 ≤ innerCandidates455_3.length then search edge 3 [3251, 4409] innerCandidates455_3 7660 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan455_4

def innerTail455_2 : List Nat := 4253 :: innerTail455_3
def innerCandidates455_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates455_2_checked : candidates edge 4 3251 4253 innerTail455_3 best.bound = innerCandidates455_2 := by
  unfold candidates
  have row : edge 4253 = SearchData.row4253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan455_2 : scan edge (search edge 3) 4 [3251] 3251 innerTail455_2 best = best := by
  rw [innerTail455_2, scan, if_neg (by decide), innerCandidates455_2_checked]
  change scan edge (search edge 3) 4 [3251] 3251 innerTail455_3
    (if 3 ≤ innerCandidates455_2.length then search edge 3 [3251, 4253] innerCandidates455_2 7504 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan455_3

def innerTail455_1 : List Nat := 4241 :: innerTail455_2
def innerCandidates455_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates455_1_checked : candidates edge 4 3251 4241 innerTail455_2 best.bound = innerCandidates455_1 := by
  unfold candidates
  have row : edge 4241 = SearchData.row4241.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan455_1 : scan edge (search edge 3) 4 [3251] 3251 innerTail455_1 best = best := by
  rw [innerTail455_1, scan, if_neg (by decide), innerCandidates455_1_checked]
  change scan edge (search edge 3) 4 [3251] 3251 innerTail455_2
    (if 3 ≤ innerCandidates455_1.length then search edge 3 [3251, 4241] innerCandidates455_1 7492 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan455_2

def innerTail455_0 : List Nat := 3413 :: innerTail455_1
def innerCandidates455_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates455_0_checked : candidates edge 4 3251 3413 innerTail455_1 best.bound = innerCandidates455_0 := by
  unfold candidates
  have row : edge 3413 = SearchData.row3413.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan455_0 : scan edge (search edge 3) 4 [3251] 3251 innerTail455_0 best = best := by
  rw [innerTail455_0, scan, if_neg (by decide), innerCandidates455_0_checked]
  change scan edge (search edge 3) 4 [3251] 3251 innerTail455_1
    (if 3 ≤ innerCandidates455_0.length then search edge 3 [3251, 3413] innerCandidates455_0 6664 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan455_1

theorem rootBranch455 : search edge 4 [3251] rootCandidates455 3251 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3251] 3251 innerTail455_0 best = best
  exact innerScan455_0

def rootCandidates454 : List Nat := [3463, 3529, 4057, 4153, 4621, 4789, 4993, 5023, 5581, 6007, 6553, 6673, 6703, 6841, 6883, 6997, 8191, 8293]
set_option maxRecDepth 4096 in
theorem rootCandidates454_checked : candidates edge 5 0 3229 tail455 best.bound = rootCandidates454 := by
  unfold candidates
  have row : edge 3229 = SearchData.row3229.contains := by funext q; rfl
  rw [row]
  decide

def innerTail454_18 : List Nat := []
theorem innerScan454_18 : scan edge (search edge 3) 4 [3229] 3229 innerTail454_18 best = best := rfl

def innerTail454_17 : List Nat := 8293 :: innerTail454_18
def innerCandidates454_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates454_17_checked : candidates edge 4 3229 8293 innerTail454_18 best.bound = innerCandidates454_17 := by
  unfold candidates
  have row : edge 8293 = SearchData.row8293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan454_17 : scan edge (search edge 3) 4 [3229] 3229 innerTail454_17 best = best := by
  rw [innerTail454_17, scan, if_neg (by decide), innerCandidates454_17_checked]
  change scan edge (search edge 3) 4 [3229] 3229 innerTail454_18
    (if 3 ≤ innerCandidates454_17.length then search edge 3 [3229, 8293] innerCandidates454_17 11522 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan454_18

def innerTail454_16 : List Nat := 8191 :: innerTail454_17
def innerCandidates454_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates454_16_checked : candidates edge 4 3229 8191 innerTail454_17 best.bound = innerCandidates454_16 := by
  unfold candidates
  have row : edge 8191 = SearchData.row8191.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan454_16 : scan edge (search edge 3) 4 [3229] 3229 innerTail454_16 best = best := by
  rw [innerTail454_16, scan, if_neg (by decide), innerCandidates454_16_checked]
  change scan edge (search edge 3) 4 [3229] 3229 innerTail454_17
    (if 3 ≤ innerCandidates454_16.length then search edge 3 [3229, 8191] innerCandidates454_16 11420 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan454_17

def innerTail454_15 : List Nat := 6997 :: innerTail454_16
def innerCandidates454_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates454_15_checked : candidates edge 4 3229 6997 innerTail454_16 best.bound = innerCandidates454_15 := by
  unfold candidates
  have row : edge 6997 = SearchData.row6997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan454_15 : scan edge (search edge 3) 4 [3229] 3229 innerTail454_15 best = best := by
  rw [innerTail454_15, scan, if_neg (by decide), innerCandidates454_15_checked]
  change scan edge (search edge 3) 4 [3229] 3229 innerTail454_16
    (if 3 ≤ innerCandidates454_15.length then search edge 3 [3229, 6997] innerCandidates454_15 10226 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan454_16

def innerTail454_14 : List Nat := 6883 :: innerTail454_15
def innerCandidates454_14 : List Nat := [8293]
set_option maxRecDepth 4096 in
theorem innerCandidates454_14_checked : candidates edge 4 3229 6883 innerTail454_15 best.bound = innerCandidates454_14 := by
  unfold candidates
  have row : edge 6883 = SearchData.row6883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan454_14 : scan edge (search edge 3) 4 [3229] 3229 innerTail454_14 best = best := by
  rw [innerTail454_14, scan, if_neg (by decide), innerCandidates454_14_checked]
  change scan edge (search edge 3) 4 [3229] 3229 innerTail454_15
    (if 3 ≤ innerCandidates454_14.length then search edge 3 [3229, 6883] innerCandidates454_14 10112 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan454_15

def innerTail454_13 : List Nat := 6841 :: innerTail454_14
def innerCandidates454_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates454_13_checked : candidates edge 4 3229 6841 innerTail454_14 best.bound = innerCandidates454_13 := by
  unfold candidates
  have row : edge 6841 = SearchData.row6841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan454_13 : scan edge (search edge 3) 4 [3229] 3229 innerTail454_13 best = best := by
  rw [innerTail454_13, scan, if_neg (by decide), innerCandidates454_13_checked]
  change scan edge (search edge 3) 4 [3229] 3229 innerTail454_14
    (if 3 ≤ innerCandidates454_13.length then search edge 3 [3229, 6841] innerCandidates454_13 10070 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan454_14

def innerTail454_12 : List Nat := 6703 :: innerTail454_13
def innerCandidates454_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates454_12_checked : candidates edge 4 3229 6703 innerTail454_13 best.bound = innerCandidates454_12 := by
  unfold candidates
  have row : edge 6703 = SearchData.row6703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan454_12 : scan edge (search edge 3) 4 [3229] 3229 innerTail454_12 best = best := by
  rw [innerTail454_12, scan, if_neg (by decide), innerCandidates454_12_checked]
  change scan edge (search edge 3) 4 [3229] 3229 innerTail454_13
    (if 3 ≤ innerCandidates454_12.length then search edge 3 [3229, 6703] innerCandidates454_12 9932 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan454_13

def innerTail454_11 : List Nat := 6673 :: innerTail454_12
def innerCandidates454_11 : List Nat := [8293]
set_option maxRecDepth 4096 in
theorem innerCandidates454_11_checked : candidates edge 4 3229 6673 innerTail454_12 best.bound = innerCandidates454_11 := by
  unfold candidates
  have row : edge 6673 = SearchData.row6673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan454_11 : scan edge (search edge 3) 4 [3229] 3229 innerTail454_11 best = best := by
  rw [innerTail454_11, scan, if_neg (by decide), innerCandidates454_11_checked]
  change scan edge (search edge 3) 4 [3229] 3229 innerTail454_12
    (if 3 ≤ innerCandidates454_11.length then search edge 3 [3229, 6673] innerCandidates454_11 9902 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan454_12

def innerTail454_10 : List Nat := 6553 :: innerTail454_11
def innerCandidates454_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates454_10_checked : candidates edge 4 3229 6553 innerTail454_11 best.bound = innerCandidates454_10 := by
  unfold candidates
  have row : edge 6553 = SearchData.row6553.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan454_10 : scan edge (search edge 3) 4 [3229] 3229 innerTail454_10 best = best := by
  rw [innerTail454_10, scan, if_neg (by decide), innerCandidates454_10_checked]
  change scan edge (search edge 3) 4 [3229] 3229 innerTail454_11
    (if 3 ≤ innerCandidates454_10.length then search edge 3 [3229, 6553] innerCandidates454_10 9782 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan454_11

def innerTail454_9 : List Nat := 6007 :: innerTail454_10
def innerCandidates454_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates454_9_checked : candidates edge 4 3229 6007 innerTail454_10 best.bound = innerCandidates454_9 := by
  unfold candidates
  have row : edge 6007 = SearchData.row6007.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan454_9 : scan edge (search edge 3) 4 [3229] 3229 innerTail454_9 best = best := by
  rw [innerTail454_9, scan, if_neg (by decide), innerCandidates454_9_checked]
  change scan edge (search edge 3) 4 [3229] 3229 innerTail454_10
    (if 3 ≤ innerCandidates454_9.length then search edge 3 [3229, 6007] innerCandidates454_9 9236 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan454_10

def innerTail454_8 : List Nat := 5581 :: innerTail454_9
def innerCandidates454_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates454_8_checked : candidates edge 4 3229 5581 innerTail454_9 best.bound = innerCandidates454_8 := by
  unfold candidates
  have row : edge 5581 = SearchData.row5581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan454_8 : scan edge (search edge 3) 4 [3229] 3229 innerTail454_8 best = best := by
  rw [innerTail454_8, scan, if_neg (by decide), innerCandidates454_8_checked]
  change scan edge (search edge 3) 4 [3229] 3229 innerTail454_9
    (if 3 ≤ innerCandidates454_8.length then search edge 3 [3229, 5581] innerCandidates454_8 8810 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan454_9

def innerTail454_7 : List Nat := 5023 :: innerTail454_8
def innerCandidates454_7 : List Nat := [6841]
set_option maxRecDepth 4096 in
theorem innerCandidates454_7_checked : candidates edge 4 3229 5023 innerTail454_8 best.bound = innerCandidates454_7 := by
  unfold candidates
  have row : edge 5023 = SearchData.row5023.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan454_7 : scan edge (search edge 3) 4 [3229] 3229 innerTail454_7 best = best := by
  rw [innerTail454_7, scan, if_neg (by decide), innerCandidates454_7_checked]
  change scan edge (search edge 3) 4 [3229] 3229 innerTail454_8
    (if 3 ≤ innerCandidates454_7.length then search edge 3 [3229, 5023] innerCandidates454_7 8252 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan454_8

def innerTail454_6 : List Nat := 4993 :: innerTail454_7
def innerCandidates454_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates454_6_checked : candidates edge 4 3229 4993 innerTail454_7 best.bound = innerCandidates454_6 := by
  unfold candidates
  have row : edge 4993 = SearchData.row4993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan454_6 : scan edge (search edge 3) 4 [3229] 3229 innerTail454_6 best = best := by
  rw [innerTail454_6, scan, if_neg (by decide), innerCandidates454_6_checked]
  change scan edge (search edge 3) 4 [3229] 3229 innerTail454_7
    (if 3 ≤ innerCandidates454_6.length then search edge 3 [3229, 4993] innerCandidates454_6 8222 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan454_7

def innerTail454_5 : List Nat := 4789 :: innerTail454_6
def innerCandidates454_5 : List Nat := [6553]
set_option maxRecDepth 4096 in
theorem innerCandidates454_5_checked : candidates edge 4 3229 4789 innerTail454_6 best.bound = innerCandidates454_5 := by
  unfold candidates
  have row : edge 4789 = SearchData.row4789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan454_5 : scan edge (search edge 3) 4 [3229] 3229 innerTail454_5 best = best := by
  rw [innerTail454_5, scan, if_neg (by decide), innerCandidates454_5_checked]
  change scan edge (search edge 3) 4 [3229] 3229 innerTail454_6
    (if 3 ≤ innerCandidates454_5.length then search edge 3 [3229, 4789] innerCandidates454_5 8018 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan454_6

def innerTail454_4 : List Nat := 4621 :: innerTail454_5
def innerCandidates454_4 : List Nat := [6673]
set_option maxRecDepth 4096 in
theorem innerCandidates454_4_checked : candidates edge 4 3229 4621 innerTail454_5 best.bound = innerCandidates454_4 := by
  unfold candidates
  have row : edge 4621 = SearchData.row4621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan454_4 : scan edge (search edge 3) 4 [3229] 3229 innerTail454_4 best = best := by
  rw [innerTail454_4, scan, if_neg (by decide), innerCandidates454_4_checked]
  change scan edge (search edge 3) 4 [3229] 3229 innerTail454_5
    (if 3 ≤ innerCandidates454_4.length then search edge 3 [3229, 4621] innerCandidates454_4 7850 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan454_5

def innerTail454_3 : List Nat := 4153 :: innerTail454_4
def innerCandidates454_3 : List Nat := [4789, 5023, 6997]
set_option maxRecDepth 4096 in
theorem innerCandidates454_3_checked : candidates edge 4 3229 4153 innerTail454_4 best.bound = innerCandidates454_3 := by
  unfold candidates
  have row : edge 4153 = SearchData.row4153.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild454_3 : search edge 3 [3229, 4153] innerCandidates454_3 7382 best = best := by decide

theorem innerScan454_3 : scan edge (search edge 3) 4 [3229] 3229 innerTail454_3 best = best := by
  rw [innerTail454_3, scan, if_neg (by decide), innerCandidates454_3_checked]
  change scan edge (search edge 3) 4 [3229] 3229 innerTail454_4
    (if 3 ≤ innerCandidates454_3.length then search edge 3 [3229, 4153] innerCandidates454_3 7382 best else best) = best
  rw [if_pos (by decide), innerChild454_3]
  exact innerScan454_4

def innerTail454_2 : List Nat := 4057 :: innerTail454_3
def innerCandidates454_2 : List Nat := [6007]
set_option maxRecDepth 4096 in
theorem innerCandidates454_2_checked : candidates edge 4 3229 4057 innerTail454_3 best.bound = innerCandidates454_2 := by
  unfold candidates
  have row : edge 4057 = SearchData.row4057.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan454_2 : scan edge (search edge 3) 4 [3229] 3229 innerTail454_2 best = best := by
  rw [innerTail454_2, scan, if_neg (by decide), innerCandidates454_2_checked]
  change scan edge (search edge 3) 4 [3229] 3229 innerTail454_3
    (if 3 ≤ innerCandidates454_2.length then search edge 3 [3229, 4057] innerCandidates454_2 7286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan454_3

def innerTail454_1 : List Nat := 3529 :: innerTail454_2
def innerCandidates454_1 : List Nat := [6553]
set_option maxRecDepth 4096 in
theorem innerCandidates454_1_checked : candidates edge 4 3229 3529 innerTail454_2 best.bound = innerCandidates454_1 := by
  unfold candidates
  have row : edge 3529 = SearchData.row3529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan454_1 : scan edge (search edge 3) 4 [3229] 3229 innerTail454_1 best = best := by
  rw [innerTail454_1, scan, if_neg (by decide), innerCandidates454_1_checked]
  change scan edge (search edge 3) 4 [3229] 3229 innerTail454_2
    (if 3 ≤ innerCandidates454_1.length then search edge 3 [3229, 3529] innerCandidates454_1 6758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan454_2

def innerTail454_0 : List Nat := 3463 :: innerTail454_1
def innerCandidates454_0 : List Nat := [6883]
set_option maxRecDepth 4096 in
theorem innerCandidates454_0_checked : candidates edge 4 3229 3463 innerTail454_1 best.bound = innerCandidates454_0 := by
  unfold candidates
  have row : edge 3463 = SearchData.row3463.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan454_0 : scan edge (search edge 3) 4 [3229] 3229 innerTail454_0 best = best := by
  rw [innerTail454_0, scan, if_neg (by decide), innerCandidates454_0_checked]
  change scan edge (search edge 3) 4 [3229] 3229 innerTail454_1
    (if 3 ≤ innerCandidates454_0.length then search edge 3 [3229, 3463] innerCandidates454_0 6692 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan454_1

theorem rootBranch454 : search edge 4 [3229] rootCandidates454 3229 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3229] 3229 innerTail454_0 best = best
  exact innerScan454_0

def rootCandidates453 : List Nat := [3761, 3767, 3851, 4349, 4409, 4547, 4817, 6269, 6311, 7247, 7643, 7727]
set_option maxRecDepth 4096 in
theorem rootCandidates453_checked : candidates edge 5 0 3221 tail454 best.bound = rootCandidates453 := by
  unfold candidates
  have row : edge 3221 = SearchData.row3221.contains := by funext q; rfl
  rw [row]
  decide

def innerTail453_12 : List Nat := []
theorem innerScan453_12 : scan edge (search edge 3) 4 [3221] 3221 innerTail453_12 best = best := rfl

def innerTail453_11 : List Nat := 7727 :: innerTail453_12
def innerCandidates453_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates453_11_checked : candidates edge 4 3221 7727 innerTail453_12 best.bound = innerCandidates453_11 := by
  unfold candidates
  have row : edge 7727 = SearchData.row7727.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan453_11 : scan edge (search edge 3) 4 [3221] 3221 innerTail453_11 best = best := by
  rw [innerTail453_11, scan, if_neg (by decide), innerCandidates453_11_checked]
  change scan edge (search edge 3) 4 [3221] 3221 innerTail453_12
    (if 3 ≤ innerCandidates453_11.length then search edge 3 [3221, 7727] innerCandidates453_11 10948 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan453_12

def innerTail453_10 : List Nat := 7643 :: innerTail453_11
def innerCandidates453_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates453_10_checked : candidates edge 4 3221 7643 innerTail453_11 best.bound = innerCandidates453_10 := by
  unfold candidates
  have row : edge 7643 = SearchData.row7643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan453_10 : scan edge (search edge 3) 4 [3221] 3221 innerTail453_10 best = best := by
  rw [innerTail453_10, scan, if_neg (by decide), innerCandidates453_10_checked]
  change scan edge (search edge 3) 4 [3221] 3221 innerTail453_11
    (if 3 ≤ innerCandidates453_10.length then search edge 3 [3221, 7643] innerCandidates453_10 10864 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan453_11

def innerTail453_9 : List Nat := 7247 :: innerTail453_10
def innerCandidates453_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates453_9_checked : candidates edge 4 3221 7247 innerTail453_10 best.bound = innerCandidates453_9 := by
  unfold candidates
  have row : edge 7247 = SearchData.row7247.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan453_9 : scan edge (search edge 3) 4 [3221] 3221 innerTail453_9 best = best := by
  rw [innerTail453_9, scan, if_neg (by decide), innerCandidates453_9_checked]
  change scan edge (search edge 3) 4 [3221] 3221 innerTail453_10
    (if 3 ≤ innerCandidates453_9.length then search edge 3 [3221, 7247] innerCandidates453_9 10468 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan453_10

def innerTail453_8 : List Nat := 6311 :: innerTail453_9
def innerCandidates453_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates453_8_checked : candidates edge 4 3221 6311 innerTail453_9 best.bound = innerCandidates453_8 := by
  unfold candidates
  have row : edge 6311 = SearchData.row6311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan453_8 : scan edge (search edge 3) 4 [3221] 3221 innerTail453_8 best = best := by
  rw [innerTail453_8, scan, if_neg (by decide), innerCandidates453_8_checked]
  change scan edge (search edge 3) 4 [3221] 3221 innerTail453_9
    (if 3 ≤ innerCandidates453_8.length then search edge 3 [3221, 6311] innerCandidates453_8 9532 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan453_9

def innerTail453_7 : List Nat := 6269 :: innerTail453_8
def innerCandidates453_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates453_7_checked : candidates edge 4 3221 6269 innerTail453_8 best.bound = innerCandidates453_7 := by
  unfold candidates
  have row : edge 6269 = SearchData.row6269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan453_7 : scan edge (search edge 3) 4 [3221] 3221 innerTail453_7 best = best := by
  rw [innerTail453_7, scan, if_neg (by decide), innerCandidates453_7_checked]
  change scan edge (search edge 3) 4 [3221] 3221 innerTail453_8
    (if 3 ≤ innerCandidates453_7.length then search edge 3 [3221, 6269] innerCandidates453_7 9490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan453_8

def innerTail453_6 : List Nat := 4817 :: innerTail453_7
def innerCandidates453_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates453_6_checked : candidates edge 4 3221 4817 innerTail453_7 best.bound = innerCandidates453_6 := by
  unfold candidates
  have row : edge 4817 = SearchData.row4817.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan453_6 : scan edge (search edge 3) 4 [3221] 3221 innerTail453_6 best = best := by
  rw [innerTail453_6, scan, if_neg (by decide), innerCandidates453_6_checked]
  change scan edge (search edge 3) 4 [3221] 3221 innerTail453_7
    (if 3 ≤ innerCandidates453_6.length then search edge 3 [3221, 4817] innerCandidates453_6 8038 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan453_7

def innerTail453_5 : List Nat := 4547 :: innerTail453_6
def innerCandidates453_5 : List Nat := [6311]
set_option maxRecDepth 4096 in
theorem innerCandidates453_5_checked : candidates edge 4 3221 4547 innerTail453_6 best.bound = innerCandidates453_5 := by
  unfold candidates
  have row : edge 4547 = SearchData.row4547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan453_5 : scan edge (search edge 3) 4 [3221] 3221 innerTail453_5 best = best := by
  rw [innerTail453_5, scan, if_neg (by decide), innerCandidates453_5_checked]
  change scan edge (search edge 3) 4 [3221] 3221 innerTail453_6
    (if 3 ≤ innerCandidates453_5.length then search edge 3 [3221, 4547] innerCandidates453_5 7768 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan453_6

def innerTail453_4 : List Nat := 4409 :: innerTail453_5
def innerCandidates453_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates453_4_checked : candidates edge 4 3221 4409 innerTail453_5 best.bound = innerCandidates453_4 := by
  unfold candidates
  have row : edge 4409 = SearchData.row4409.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan453_4 : scan edge (search edge 3) 4 [3221] 3221 innerTail453_4 best = best := by
  rw [innerTail453_4, scan, if_neg (by decide), innerCandidates453_4_checked]
  change scan edge (search edge 3) 4 [3221] 3221 innerTail453_5
    (if 3 ≤ innerCandidates453_4.length then search edge 3 [3221, 4409] innerCandidates453_4 7630 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan453_5

def innerTail453_3 : List Nat := 4349 :: innerTail453_4
def innerCandidates453_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates453_3_checked : candidates edge 4 3221 4349 innerTail453_4 best.bound = innerCandidates453_3 := by
  unfold candidates
  have row : edge 4349 = SearchData.row4349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan453_3 : scan edge (search edge 3) 4 [3221] 3221 innerTail453_3 best = best := by
  rw [innerTail453_3, scan, if_neg (by decide), innerCandidates453_3_checked]
  change scan edge (search edge 3) 4 [3221] 3221 innerTail453_4
    (if 3 ≤ innerCandidates453_3.length then search edge 3 [3221, 4349] innerCandidates453_3 7570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan453_4

def innerTail453_2 : List Nat := 3851 :: innerTail453_3
def innerCandidates453_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates453_2_checked : candidates edge 4 3221 3851 innerTail453_3 best.bound = innerCandidates453_2 := by
  unfold candidates
  have row : edge 3851 = SearchData.row3851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan453_2 : scan edge (search edge 3) 4 [3221] 3221 innerTail453_2 best = best := by
  rw [innerTail453_2, scan, if_neg (by decide), innerCandidates453_2_checked]
  change scan edge (search edge 3) 4 [3221] 3221 innerTail453_3
    (if 3 ≤ innerCandidates453_2.length then search edge 3 [3221, 3851] innerCandidates453_2 7072 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan453_3

def innerTail453_1 : List Nat := 3767 :: innerTail453_2
def innerCandidates453_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates453_1_checked : candidates edge 4 3221 3767 innerTail453_2 best.bound = innerCandidates453_1 := by
  unfold candidates
  have row : edge 3767 = SearchData.row3767.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan453_1 : scan edge (search edge 3) 4 [3221] 3221 innerTail453_1 best = best := by
  rw [innerTail453_1, scan, if_neg (by decide), innerCandidates453_1_checked]
  change scan edge (search edge 3) 4 [3221] 3221 innerTail453_2
    (if 3 ≤ innerCandidates453_1.length then search edge 3 [3221, 3767] innerCandidates453_1 6988 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan453_2

def innerTail453_0 : List Nat := 3761 :: innerTail453_1
def innerCandidates453_0 : List Nat := [3767]
set_option maxRecDepth 4096 in
theorem innerCandidates453_0_checked : candidates edge 4 3221 3761 innerTail453_1 best.bound = innerCandidates453_0 := by
  unfold candidates
  have row : edge 3761 = SearchData.row3761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan453_0 : scan edge (search edge 3) 4 [3221] 3221 innerTail453_0 best = best := by
  rw [innerTail453_0, scan, if_neg (by decide), innerCandidates453_0_checked]
  change scan edge (search edge 3) 4 [3221] 3221 innerTail453_1
    (if 3 ≤ innerCandidates453_0.length then search edge 3 [3221, 3761] innerCandidates453_0 6982 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan453_1

theorem rootBranch453 : search edge 4 [3221] rootCandidates453 3221 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3221] 3221 innerTail453_0 best = best
  exact innerScan453_0

end ProjectEulerSolutions.PrimePairUpper.Search
