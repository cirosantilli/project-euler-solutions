import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates179 : List Nat := [1319, 1697, 1847, 1889, 1949, 2267, 2633, 3041, 3527, 4373, 4919, 4931, 7307, 8081, 8111]
set_option maxRecDepth 4096 in
theorem rootCandidates179_checked : candidates edge 5 0 1091 tail180 best.bound = rootCandidates179 := by
  unfold candidates
  have row : edge 1091 = PrimePairCache.smallRow1091.contains := by funext q; rfl
  rw [row]
  decide

def innerTail179_15 : List Nat := []
theorem innerScan179_15 : scan edge (search edge 3) 4 [1091] 1091 innerTail179_15 best = best := rfl

def innerTail179_14 : List Nat := 8111 :: innerTail179_15
def innerCandidates179_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates179_14_checked : candidates edge 4 1091 8111 innerTail179_15 best.bound = innerCandidates179_14 := by
  unfold candidates
  have row : edge 8111 = SearchData.row8111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan179_14 : scan edge (search edge 3) 4 [1091] 1091 innerTail179_14 best = best := by
  rw [innerTail179_14, scan, if_neg (by decide), innerCandidates179_14_checked]
  change scan edge (search edge 3) 4 [1091] 1091 innerTail179_15
    (if 3 ≤ innerCandidates179_14.length then search edge 3 [1091, 8111] innerCandidates179_14 9202 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan179_15

def innerTail179_13 : List Nat := 8081 :: innerTail179_14
def innerCandidates179_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates179_13_checked : candidates edge 4 1091 8081 innerTail179_14 best.bound = innerCandidates179_13 := by
  unfold candidates
  have row : edge 8081 = SearchData.row8081.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan179_13 : scan edge (search edge 3) 4 [1091] 1091 innerTail179_13 best = best := by
  rw [innerTail179_13, scan, if_neg (by decide), innerCandidates179_13_checked]
  change scan edge (search edge 3) 4 [1091] 1091 innerTail179_14
    (if 3 ≤ innerCandidates179_13.length then search edge 3 [1091, 8081] innerCandidates179_13 9172 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan179_14

def innerTail179_12 : List Nat := 7307 :: innerTail179_13
def innerCandidates179_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates179_12_checked : candidates edge 4 1091 7307 innerTail179_13 best.bound = innerCandidates179_12 := by
  unfold candidates
  have row : edge 7307 = SearchData.row7307.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan179_12 : scan edge (search edge 3) 4 [1091] 1091 innerTail179_12 best = best := by
  rw [innerTail179_12, scan, if_neg (by decide), innerCandidates179_12_checked]
  change scan edge (search edge 3) 4 [1091] 1091 innerTail179_13
    (if 3 ≤ innerCandidates179_12.length then search edge 3 [1091, 7307] innerCandidates179_12 8398 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan179_13

def innerTail179_11 : List Nat := 4931 :: innerTail179_12
def innerCandidates179_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates179_11_checked : candidates edge 4 1091 4931 innerTail179_12 best.bound = innerCandidates179_11 := by
  unfold candidates
  have row : edge 4931 = SearchData.row4931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan179_11 : scan edge (search edge 3) 4 [1091] 1091 innerTail179_11 best = best := by
  rw [innerTail179_11, scan, if_neg (by decide), innerCandidates179_11_checked]
  change scan edge (search edge 3) 4 [1091] 1091 innerTail179_12
    (if 3 ≤ innerCandidates179_11.length then search edge 3 [1091, 4931] innerCandidates179_11 6022 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan179_12

def innerTail179_10 : List Nat := 4919 :: innerTail179_11
def innerCandidates179_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates179_10_checked : candidates edge 4 1091 4919 innerTail179_11 best.bound = innerCandidates179_10 := by
  unfold candidates
  have row : edge 4919 = SearchData.row4919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan179_10 : scan edge (search edge 3) 4 [1091] 1091 innerTail179_10 best = best := by
  rw [innerTail179_10, scan, if_neg (by decide), innerCandidates179_10_checked]
  change scan edge (search edge 3) 4 [1091] 1091 innerTail179_11
    (if 3 ≤ innerCandidates179_10.length then search edge 3 [1091, 4919] innerCandidates179_10 6010 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan179_11

def innerTail179_9 : List Nat := 4373 :: innerTail179_10
def innerCandidates179_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates179_9_checked : candidates edge 4 1091 4373 innerTail179_10 best.bound = innerCandidates179_9 := by
  unfold candidates
  have row : edge 4373 = SearchData.row4373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan179_9 : scan edge (search edge 3) 4 [1091] 1091 innerTail179_9 best = best := by
  rw [innerTail179_9, scan, if_neg (by decide), innerCandidates179_9_checked]
  change scan edge (search edge 3) 4 [1091] 1091 innerTail179_10
    (if 3 ≤ innerCandidates179_9.length then search edge 3 [1091, 4373] innerCandidates179_9 5464 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan179_10

def innerTail179_8 : List Nat := 3527 :: innerTail179_9
def innerCandidates179_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates179_8_checked : candidates edge 4 1091 3527 innerTail179_9 best.bound = innerCandidates179_8 := by
  unfold candidates
  have row : edge 3527 = SearchData.row3527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan179_8 : scan edge (search edge 3) 4 [1091] 1091 innerTail179_8 best = best := by
  rw [innerTail179_8, scan, if_neg (by decide), innerCandidates179_8_checked]
  change scan edge (search edge 3) 4 [1091] 1091 innerTail179_9
    (if 3 ≤ innerCandidates179_8.length then search edge 3 [1091, 3527] innerCandidates179_8 4618 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan179_9

def innerTail179_7 : List Nat := 3041 :: innerTail179_8
def innerCandidates179_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates179_7_checked : candidates edge 4 1091 3041 innerTail179_8 best.bound = innerCandidates179_7 := by
  unfold candidates
  have row : edge 3041 = SearchData.row3041.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan179_7 : scan edge (search edge 3) 4 [1091] 1091 innerTail179_7 best = best := by
  rw [innerTail179_7, scan, if_neg (by decide), innerCandidates179_7_checked]
  change scan edge (search edge 3) 4 [1091] 1091 innerTail179_8
    (if 3 ≤ innerCandidates179_7.length then search edge 3 [1091, 3041] innerCandidates179_7 4132 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan179_8

def innerTail179_6 : List Nat := 2633 :: innerTail179_7
def innerCandidates179_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates179_6_checked : candidates edge 4 1091 2633 innerTail179_7 best.bound = innerCandidates179_6 := by
  unfold candidates
  have row : edge 2633 = PrimePairCache.smallRow2633.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan179_6 : scan edge (search edge 3) 4 [1091] 1091 innerTail179_6 best = best := by
  rw [innerTail179_6, scan, if_neg (by decide), innerCandidates179_6_checked]
  change scan edge (search edge 3) 4 [1091] 1091 innerTail179_7
    (if 3 ≤ innerCandidates179_6.length then search edge 3 [1091, 2633] innerCandidates179_6 3724 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan179_7

def innerTail179_5 : List Nat := 2267 :: innerTail179_6
def innerCandidates179_5 : List Nat := [3527]
set_option maxRecDepth 4096 in
theorem innerCandidates179_5_checked : candidates edge 4 1091 2267 innerTail179_6 best.bound = innerCandidates179_5 := by
  unfold candidates
  have row : edge 2267 = PrimePairCache.smallRow2267.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan179_5 : scan edge (search edge 3) 4 [1091] 1091 innerTail179_5 best = best := by
  rw [innerTail179_5, scan, if_neg (by decide), innerCandidates179_5_checked]
  change scan edge (search edge 3) 4 [1091] 1091 innerTail179_6
    (if 3 ≤ innerCandidates179_5.length then search edge 3 [1091, 2267] innerCandidates179_5 3358 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan179_6

def innerTail179_4 : List Nat := 1949 :: innerTail179_5
def innerCandidates179_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates179_4_checked : candidates edge 4 1091 1949 innerTail179_5 best.bound = innerCandidates179_4 := by
  unfold candidates
  have row : edge 1949 = PrimePairCache.smallRow1949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan179_4 : scan edge (search edge 3) 4 [1091] 1091 innerTail179_4 best = best := by
  rw [innerTail179_4, scan, if_neg (by decide), innerCandidates179_4_checked]
  change scan edge (search edge 3) 4 [1091] 1091 innerTail179_5
    (if 3 ≤ innerCandidates179_4.length then search edge 3 [1091, 1949] innerCandidates179_4 3040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan179_5

def innerTail179_3 : List Nat := 1889 :: innerTail179_4
def innerCandidates179_3 : List Nat := [2633, 4919]
set_option maxRecDepth 4096 in
theorem innerCandidates179_3_checked : candidates edge 4 1091 1889 innerTail179_4 best.bound = innerCandidates179_3 := by
  unfold candidates
  have row : edge 1889 = PrimePairCache.smallRow1889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan179_3 : scan edge (search edge 3) 4 [1091] 1091 innerTail179_3 best = best := by
  rw [innerTail179_3, scan, if_neg (by decide), innerCandidates179_3_checked]
  change scan edge (search edge 3) 4 [1091] 1091 innerTail179_4
    (if 3 ≤ innerCandidates179_3.length then search edge 3 [1091, 1889] innerCandidates179_3 2980 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan179_4

def innerTail179_2 : List Nat := 1847 :: innerTail179_3
def innerCandidates179_2 : List Nat := [4373, 4919, 7307]
set_option maxRecDepth 4096 in
theorem innerCandidates179_2_checked : candidates edge 4 1091 1847 innerTail179_3 best.bound = innerCandidates179_2 := by
  unfold candidates
  have row : edge 1847 = PrimePairCache.smallRow1847.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild179_2 : search edge 3 [1091, 1847] innerCandidates179_2 2938 best = best := by decide

theorem innerScan179_2 : scan edge (search edge 3) 4 [1091] 1091 innerTail179_2 best = best := by
  rw [innerTail179_2, scan, if_neg (by decide), innerCandidates179_2_checked]
  change scan edge (search edge 3) 4 [1091] 1091 innerTail179_3
    (if 3 ≤ innerCandidates179_2.length then search edge 3 [1091, 1847] innerCandidates179_2 2938 best else best) = best
  rw [if_pos (by decide), innerChild179_2]
  exact innerScan179_3

def innerTail179_1 : List Nat := 1697 :: innerTail179_2
def innerCandidates179_1 : List Nat := [3041, 3527, 4931]
set_option maxRecDepth 4096 in
theorem innerCandidates179_1_checked : candidates edge 4 1091 1697 innerTail179_2 best.bound = innerCandidates179_1 := by
  unfold candidates
  have row : edge 1697 = PrimePairCache.smallRow1697.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild179_1 : search edge 3 [1091, 1697] innerCandidates179_1 2788 best = best := by decide

theorem innerScan179_1 : scan edge (search edge 3) 4 [1091] 1091 innerTail179_1 best = best := by
  rw [innerTail179_1, scan, if_neg (by decide), innerCandidates179_1_checked]
  change scan edge (search edge 3) 4 [1091] 1091 innerTail179_2
    (if 3 ≤ innerCandidates179_1.length then search edge 3 [1091, 1697] innerCandidates179_1 2788 best else best) = best
  rw [if_pos (by decide), innerChild179_1]
  exact innerScan179_2

def innerTail179_0 : List Nat := 1319 :: innerTail179_1
def innerCandidates179_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates179_0_checked : candidates edge 4 1091 1319 innerTail179_1 best.bound = innerCandidates179_0 := by
  unfold candidates
  have row : edge 1319 = PrimePairCache.smallRow1319.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan179_0 : scan edge (search edge 3) 4 [1091] 1091 innerTail179_0 best = best := by
  rw [innerTail179_0, scan, if_neg (by decide), innerCandidates179_0_checked]
  change scan edge (search edge 3) 4 [1091] 1091 innerTail179_1
    (if 3 ≤ innerCandidates179_0.length then search edge 3 [1091, 1319] innerCandidates179_0 2410 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan179_1

theorem rootBranch179 : search edge 4 [1091] rootCandidates179 1091 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1091] 1091 innerTail179_0 best = best
  exact innerScan179_0

def rootCandidates178 : List Nat := [1117, 1453, 2137, 2179, 2833, 3271, 3847, 4447, 4789, 4957, 5107, 5413, 5569, 6091, 6451, 6763, 6967]
set_option maxRecDepth 4096 in
theorem rootCandidates178_checked : candidates edge 5 0 1087 tail179 best.bound = rootCandidates178 := by
  unfold candidates
  have row : edge 1087 = PrimePairCache.smallRow1087.contains := by funext q; rfl
  rw [row]
  decide

def innerTail178_17 : List Nat := []
theorem innerScan178_17 : scan edge (search edge 3) 4 [1087] 1087 innerTail178_17 best = best := rfl

def innerTail178_16 : List Nat := 6967 :: innerTail178_17
def innerCandidates178_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates178_16_checked : candidates edge 4 1087 6967 innerTail178_17 best.bound = innerCandidates178_16 := by
  unfold candidates
  have row : edge 6967 = SearchData.row6967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan178_16 : scan edge (search edge 3) 4 [1087] 1087 innerTail178_16 best = best := by
  rw [innerTail178_16, scan, if_neg (by decide), innerCandidates178_16_checked]
  change scan edge (search edge 3) 4 [1087] 1087 innerTail178_17
    (if 3 ≤ innerCandidates178_16.length then search edge 3 [1087, 6967] innerCandidates178_16 8054 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan178_17

def innerTail178_15 : List Nat := 6763 :: innerTail178_16
def innerCandidates178_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates178_15_checked : candidates edge 4 1087 6763 innerTail178_16 best.bound = innerCandidates178_15 := by
  unfold candidates
  have row : edge 6763 = SearchData.row6763.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan178_15 : scan edge (search edge 3) 4 [1087] 1087 innerTail178_15 best = best := by
  rw [innerTail178_15, scan, if_neg (by decide), innerCandidates178_15_checked]
  change scan edge (search edge 3) 4 [1087] 1087 innerTail178_16
    (if 3 ≤ innerCandidates178_15.length then search edge 3 [1087, 6763] innerCandidates178_15 7850 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan178_16

def innerTail178_14 : List Nat := 6451 :: innerTail178_15
def innerCandidates178_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates178_14_checked : candidates edge 4 1087 6451 innerTail178_15 best.bound = innerCandidates178_14 := by
  unfold candidates
  have row : edge 6451 = SearchData.row6451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan178_14 : scan edge (search edge 3) 4 [1087] 1087 innerTail178_14 best = best := by
  rw [innerTail178_14, scan, if_neg (by decide), innerCandidates178_14_checked]
  change scan edge (search edge 3) 4 [1087] 1087 innerTail178_15
    (if 3 ≤ innerCandidates178_14.length then search edge 3 [1087, 6451] innerCandidates178_14 7538 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan178_15

def innerTail178_13 : List Nat := 6091 :: innerTail178_14
def innerCandidates178_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates178_13_checked : candidates edge 4 1087 6091 innerTail178_14 best.bound = innerCandidates178_13 := by
  unfold candidates
  have row : edge 6091 = SearchData.row6091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan178_13 : scan edge (search edge 3) 4 [1087] 1087 innerTail178_13 best = best := by
  rw [innerTail178_13, scan, if_neg (by decide), innerCandidates178_13_checked]
  change scan edge (search edge 3) 4 [1087] 1087 innerTail178_14
    (if 3 ≤ innerCandidates178_13.length then search edge 3 [1087, 6091] innerCandidates178_13 7178 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan178_14

def innerTail178_12 : List Nat := 5569 :: innerTail178_13
def innerCandidates178_12 : List Nat := [6451]
set_option maxRecDepth 4096 in
theorem innerCandidates178_12_checked : candidates edge 4 1087 5569 innerTail178_13 best.bound = innerCandidates178_12 := by
  unfold candidates
  have row : edge 5569 = SearchData.row5569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan178_12 : scan edge (search edge 3) 4 [1087] 1087 innerTail178_12 best = best := by
  rw [innerTail178_12, scan, if_neg (by decide), innerCandidates178_12_checked]
  change scan edge (search edge 3) 4 [1087] 1087 innerTail178_13
    (if 3 ≤ innerCandidates178_12.length then search edge 3 [1087, 5569] innerCandidates178_12 6656 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan178_13

def innerTail178_11 : List Nat := 5413 :: innerTail178_12
def innerCandidates178_11 : List Nat := [5569, 6451]
set_option maxRecDepth 4096 in
theorem innerCandidates178_11_checked : candidates edge 4 1087 5413 innerTail178_12 best.bound = innerCandidates178_11 := by
  unfold candidates
  have row : edge 5413 = SearchData.row5413.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan178_11 : scan edge (search edge 3) 4 [1087] 1087 innerTail178_11 best = best := by
  rw [innerTail178_11, scan, if_neg (by decide), innerCandidates178_11_checked]
  change scan edge (search edge 3) 4 [1087] 1087 innerTail178_12
    (if 3 ≤ innerCandidates178_11.length then search edge 3 [1087, 5413] innerCandidates178_11 6500 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan178_12

def innerTail178_10 : List Nat := 5107 :: innerTail178_11
def innerCandidates178_10 : List Nat := [6763]
set_option maxRecDepth 4096 in
theorem innerCandidates178_10_checked : candidates edge 4 1087 5107 innerTail178_11 best.bound = innerCandidates178_10 := by
  unfold candidates
  have row : edge 5107 = SearchData.row5107.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan178_10 : scan edge (search edge 3) 4 [1087] 1087 innerTail178_10 best = best := by
  rw [innerTail178_10, scan, if_neg (by decide), innerCandidates178_10_checked]
  change scan edge (search edge 3) 4 [1087] 1087 innerTail178_11
    (if 3 ≤ innerCandidates178_10.length then search edge 3 [1087, 5107] innerCandidates178_10 6194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan178_11

def innerTail178_9 : List Nat := 4957 :: innerTail178_10
def innerCandidates178_9 : List Nat := [5569]
set_option maxRecDepth 4096 in
theorem innerCandidates178_9_checked : candidates edge 4 1087 4957 innerTail178_10 best.bound = innerCandidates178_9 := by
  unfold candidates
  have row : edge 4957 = SearchData.row4957.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan178_9 : scan edge (search edge 3) 4 [1087] 1087 innerTail178_9 best = best := by
  rw [innerTail178_9, scan, if_neg (by decide), innerCandidates178_9_checked]
  change scan edge (search edge 3) 4 [1087] 1087 innerTail178_10
    (if 3 ≤ innerCandidates178_9.length then search edge 3 [1087, 4957] innerCandidates178_9 6044 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan178_10

def innerTail178_8 : List Nat := 4789 :: innerTail178_9
def innerCandidates178_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates178_8_checked : candidates edge 4 1087 4789 innerTail178_9 best.bound = innerCandidates178_8 := by
  unfold candidates
  have row : edge 4789 = SearchData.row4789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan178_8 : scan edge (search edge 3) 4 [1087] 1087 innerTail178_8 best = best := by
  rw [innerTail178_8, scan, if_neg (by decide), innerCandidates178_8_checked]
  change scan edge (search edge 3) 4 [1087] 1087 innerTail178_9
    (if 3 ≤ innerCandidates178_8.length then search edge 3 [1087, 4789] innerCandidates178_8 5876 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan178_9

def innerTail178_7 : List Nat := 4447 :: innerTail178_8
def innerCandidates178_7 : List Nat := [4789]
set_option maxRecDepth 4096 in
theorem innerCandidates178_7_checked : candidates edge 4 1087 4447 innerTail178_8 best.bound = innerCandidates178_7 := by
  unfold candidates
  have row : edge 4447 = SearchData.row4447.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan178_7 : scan edge (search edge 3) 4 [1087] 1087 innerTail178_7 best = best := by
  rw [innerTail178_7, scan, if_neg (by decide), innerCandidates178_7_checked]
  change scan edge (search edge 3) 4 [1087] 1087 innerTail178_8
    (if 3 ≤ innerCandidates178_7.length then search edge 3 [1087, 4447] innerCandidates178_7 5534 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan178_8

def innerTail178_6 : List Nat := 3847 :: innerTail178_7
def innerCandidates178_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates178_6_checked : candidates edge 4 1087 3847 innerTail178_7 best.bound = innerCandidates178_6 := by
  unfold candidates
  have row : edge 3847 = SearchData.row3847.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan178_6 : scan edge (search edge 3) 4 [1087] 1087 innerTail178_6 best = best := by
  rw [innerTail178_6, scan, if_neg (by decide), innerCandidates178_6_checked]
  change scan edge (search edge 3) 4 [1087] 1087 innerTail178_7
    (if 3 ≤ innerCandidates178_6.length then search edge 3 [1087, 3847] innerCandidates178_6 4934 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan178_7

def innerTail178_5 : List Nat := 3271 :: innerTail178_6
def innerCandidates178_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates178_5_checked : candidates edge 4 1087 3271 innerTail178_6 best.bound = innerCandidates178_5 := by
  unfold candidates
  have row : edge 3271 = SearchData.row3271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan178_5 : scan edge (search edge 3) 4 [1087] 1087 innerTail178_5 best = best := by
  rw [innerTail178_5, scan, if_neg (by decide), innerCandidates178_5_checked]
  change scan edge (search edge 3) 4 [1087] 1087 innerTail178_6
    (if 3 ≤ innerCandidates178_5.length then search edge 3 [1087, 3271] innerCandidates178_5 4358 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan178_6

def innerTail178_4 : List Nat := 2833 :: innerTail178_5
def innerCandidates178_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates178_4_checked : candidates edge 4 1087 2833 innerTail178_5 best.bound = innerCandidates178_4 := by
  unfold candidates
  have row : edge 2833 = SearchData.row2833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan178_4 : scan edge (search edge 3) 4 [1087] 1087 innerTail178_4 best = best := by
  rw [innerTail178_4, scan, if_neg (by decide), innerCandidates178_4_checked]
  change scan edge (search edge 3) 4 [1087] 1087 innerTail178_5
    (if 3 ≤ innerCandidates178_4.length then search edge 3 [1087, 2833] innerCandidates178_4 3920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan178_5

def innerTail178_3 : List Nat := 2179 :: innerTail178_4
def innerCandidates178_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates178_3_checked : candidates edge 4 1087 2179 innerTail178_4 best.bound = innerCandidates178_3 := by
  unfold candidates
  have row : edge 2179 = PrimePairCache.smallRow2179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan178_3 : scan edge (search edge 3) 4 [1087] 1087 innerTail178_3 best = best := by
  rw [innerTail178_3, scan, if_neg (by decide), innerCandidates178_3_checked]
  change scan edge (search edge 3) 4 [1087] 1087 innerTail178_4
    (if 3 ≤ innerCandidates178_3.length then search edge 3 [1087, 2179] innerCandidates178_3 3266 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan178_4

def innerTail178_2 : List Nat := 2137 :: innerTail178_3
def innerCandidates178_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates178_2_checked : candidates edge 4 1087 2137 innerTail178_3 best.bound = innerCandidates178_2 := by
  unfold candidates
  have row : edge 2137 = PrimePairCache.smallRow2137.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan178_2 : scan edge (search edge 3) 4 [1087] 1087 innerTail178_2 best = best := by
  rw [innerTail178_2, scan, if_neg (by decide), innerCandidates178_2_checked]
  change scan edge (search edge 3) 4 [1087] 1087 innerTail178_3
    (if 3 ≤ innerCandidates178_2.length then search edge 3 [1087, 2137] innerCandidates178_2 3224 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan178_3

def innerTail178_1 : List Nat := 1453 :: innerTail178_2
def innerCandidates178_1 : List Nat := [6763]
set_option maxRecDepth 4096 in
theorem innerCandidates178_1_checked : candidates edge 4 1087 1453 innerTail178_2 best.bound = innerCandidates178_1 := by
  unfold candidates
  have row : edge 1453 = PrimePairCache.smallRow1453.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan178_1 : scan edge (search edge 3) 4 [1087] 1087 innerTail178_1 best = best := by
  rw [innerTail178_1, scan, if_neg (by decide), innerCandidates178_1_checked]
  change scan edge (search edge 3) 4 [1087] 1087 innerTail178_2
    (if 3 ≤ innerCandidates178_1.length then search edge 3 [1087, 1453] innerCandidates178_1 2540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan178_2

def innerTail178_0 : List Nat := 1117 :: innerTail178_1
def innerCandidates178_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates178_0_checked : candidates edge 4 1087 1117 innerTail178_1 best.bound = innerCandidates178_0 := by
  unfold candidates
  have row : edge 1117 = PrimePairCache.smallRow1117.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan178_0 : scan edge (search edge 3) 4 [1087] 1087 innerTail178_0 best = best := by
  rw [innerTail178_0, scan, if_neg (by decide), innerCandidates178_0_checked]
  change scan edge (search edge 3) 4 [1087] 1087 innerTail178_1
    (if 3 ≤ innerCandidates178_0.length then search edge 3 [1087, 1117] innerCandidates178_0 2204 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan178_1

theorem rootBranch178 : search edge 4 [1087] rootCandidates178 1087 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1087] 1087 innerTail178_0 best = best
  exact innerScan178_0

def rootCandidates177 : List Nat := [1321, 1609, 2113, 2203, 2281, 2659, 3061, 3313, 4219, 4261, 4909, 5059, 5683, 6337, 6529, 6841, 6883, 7069, 7993, 8161, 8221, 8329]
set_option maxRecDepth 4096 in
theorem rootCandidates177_checked : candidates edge 5 0 1069 tail178 best.bound = rootCandidates177 := by
  unfold candidates
  have row : edge 1069 = PrimePairCache.smallRow1069.contains := by funext q; rfl
  rw [row]
  decide

def innerTail177_22 : List Nat := []
theorem innerScan177_22 : scan edge (search edge 3) 4 [1069] 1069 innerTail177_22 best = best := rfl

def innerTail177_21 : List Nat := 8329 :: innerTail177_22
def innerCandidates177_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates177_21_checked : candidates edge 4 1069 8329 innerTail177_22 best.bound = innerCandidates177_21 := by
  unfold candidates
  have row : edge 8329 = SearchData.row8329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan177_21 : scan edge (search edge 3) 4 [1069] 1069 innerTail177_21 best = best := by
  rw [innerTail177_21, scan, if_neg (by decide), innerCandidates177_21_checked]
  change scan edge (search edge 3) 4 [1069] 1069 innerTail177_22
    (if 3 ≤ innerCandidates177_21.length then search edge 3 [1069, 8329] innerCandidates177_21 9398 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan177_22

def innerTail177_20 : List Nat := 8221 :: innerTail177_21
def innerCandidates177_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates177_20_checked : candidates edge 4 1069 8221 innerTail177_21 best.bound = innerCandidates177_20 := by
  unfold candidates
  have row : edge 8221 = SearchData.row8221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan177_20 : scan edge (search edge 3) 4 [1069] 1069 innerTail177_20 best = best := by
  rw [innerTail177_20, scan, if_neg (by decide), innerCandidates177_20_checked]
  change scan edge (search edge 3) 4 [1069] 1069 innerTail177_21
    (if 3 ≤ innerCandidates177_20.length then search edge 3 [1069, 8221] innerCandidates177_20 9290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan177_21

def innerTail177_19 : List Nat := 8161 :: innerTail177_20
def innerCandidates177_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates177_19_checked : candidates edge 4 1069 8161 innerTail177_20 best.bound = innerCandidates177_19 := by
  unfold candidates
  have row : edge 8161 = SearchData.row8161.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan177_19 : scan edge (search edge 3) 4 [1069] 1069 innerTail177_19 best = best := by
  rw [innerTail177_19, scan, if_neg (by decide), innerCandidates177_19_checked]
  change scan edge (search edge 3) 4 [1069] 1069 innerTail177_20
    (if 3 ≤ innerCandidates177_19.length then search edge 3 [1069, 8161] innerCandidates177_19 9230 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan177_20

def innerTail177_18 : List Nat := 7993 :: innerTail177_19
def innerCandidates177_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates177_18_checked : candidates edge 4 1069 7993 innerTail177_19 best.bound = innerCandidates177_18 := by
  unfold candidates
  have row : edge 7993 = SearchData.row7993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan177_18 : scan edge (search edge 3) 4 [1069] 1069 innerTail177_18 best = best := by
  rw [innerTail177_18, scan, if_neg (by decide), innerCandidates177_18_checked]
  change scan edge (search edge 3) 4 [1069] 1069 innerTail177_19
    (if 3 ≤ innerCandidates177_18.length then search edge 3 [1069, 7993] innerCandidates177_18 9062 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan177_19

def innerTail177_17 : List Nat := 7069 :: innerTail177_18
def innerCandidates177_17 : List Nat := [8161]
set_option maxRecDepth 4096 in
theorem innerCandidates177_17_checked : candidates edge 4 1069 7069 innerTail177_18 best.bound = innerCandidates177_17 := by
  unfold candidates
  have row : edge 7069 = SearchData.row7069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan177_17 : scan edge (search edge 3) 4 [1069] 1069 innerTail177_17 best = best := by
  rw [innerTail177_17, scan, if_neg (by decide), innerCandidates177_17_checked]
  change scan edge (search edge 3) 4 [1069] 1069 innerTail177_18
    (if 3 ≤ innerCandidates177_17.length then search edge 3 [1069, 7069] innerCandidates177_17 8138 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan177_18

def innerTail177_16 : List Nat := 6883 :: innerTail177_17
def innerCandidates177_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates177_16_checked : candidates edge 4 1069 6883 innerTail177_17 best.bound = innerCandidates177_16 := by
  unfold candidates
  have row : edge 6883 = SearchData.row6883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan177_16 : scan edge (search edge 3) 4 [1069] 1069 innerTail177_16 best = best := by
  rw [innerTail177_16, scan, if_neg (by decide), innerCandidates177_16_checked]
  change scan edge (search edge 3) 4 [1069] 1069 innerTail177_17
    (if 3 ≤ innerCandidates177_16.length then search edge 3 [1069, 6883] innerCandidates177_16 7952 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan177_17

def innerTail177_15 : List Nat := 6841 :: innerTail177_16
def innerCandidates177_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates177_15_checked : candidates edge 4 1069 6841 innerTail177_16 best.bound = innerCandidates177_15 := by
  unfold candidates
  have row : edge 6841 = SearchData.row6841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan177_15 : scan edge (search edge 3) 4 [1069] 1069 innerTail177_15 best = best := by
  rw [innerTail177_15, scan, if_neg (by decide), innerCandidates177_15_checked]
  change scan edge (search edge 3) 4 [1069] 1069 innerTail177_16
    (if 3 ≤ innerCandidates177_15.length then search edge 3 [1069, 6841] innerCandidates177_15 7910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan177_16

def innerTail177_14 : List Nat := 6529 :: innerTail177_15
def innerCandidates177_14 : List Nat := [8221, 8329]
set_option maxRecDepth 4096 in
theorem innerCandidates177_14_checked : candidates edge 4 1069 6529 innerTail177_15 best.bound = innerCandidates177_14 := by
  unfold candidates
  have row : edge 6529 = SearchData.row6529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan177_14 : scan edge (search edge 3) 4 [1069] 1069 innerTail177_14 best = best := by
  rw [innerTail177_14, scan, if_neg (by decide), innerCandidates177_14_checked]
  change scan edge (search edge 3) 4 [1069] 1069 innerTail177_15
    (if 3 ≤ innerCandidates177_14.length then search edge 3 [1069, 6529] innerCandidates177_14 7598 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan177_15

def innerTail177_13 : List Nat := 6337 :: innerTail177_14
def innerCandidates177_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates177_13_checked : candidates edge 4 1069 6337 innerTail177_14 best.bound = innerCandidates177_13 := by
  unfold candidates
  have row : edge 6337 = SearchData.row6337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan177_13 : scan edge (search edge 3) 4 [1069] 1069 innerTail177_13 best = best := by
  rw [innerTail177_13, scan, if_neg (by decide), innerCandidates177_13_checked]
  change scan edge (search edge 3) 4 [1069] 1069 innerTail177_14
    (if 3 ≤ innerCandidates177_13.length then search edge 3 [1069, 6337] innerCandidates177_13 7406 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan177_14

def innerTail177_12 : List Nat := 5683 :: innerTail177_13
def innerCandidates177_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates177_12_checked : candidates edge 4 1069 5683 innerTail177_13 best.bound = innerCandidates177_12 := by
  unfold candidates
  have row : edge 5683 = SearchData.row5683.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan177_12 : scan edge (search edge 3) 4 [1069] 1069 innerTail177_12 best = best := by
  rw [innerTail177_12, scan, if_neg (by decide), innerCandidates177_12_checked]
  change scan edge (search edge 3) 4 [1069] 1069 innerTail177_13
    (if 3 ≤ innerCandidates177_12.length then search edge 3 [1069, 5683] innerCandidates177_12 6752 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan177_13

def innerTail177_11 : List Nat := 5059 :: innerTail177_12
def innerCandidates177_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates177_11_checked : candidates edge 4 1069 5059 innerTail177_12 best.bound = innerCandidates177_11 := by
  unfold candidates
  have row : edge 5059 = SearchData.row5059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan177_11 : scan edge (search edge 3) 4 [1069] 1069 innerTail177_11 best = best := by
  rw [innerTail177_11, scan, if_neg (by decide), innerCandidates177_11_checked]
  change scan edge (search edge 3) 4 [1069] 1069 innerTail177_12
    (if 3 ≤ innerCandidates177_11.length then search edge 3 [1069, 5059] innerCandidates177_11 6128 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan177_12

def innerTail177_10 : List Nat := 4909 :: innerTail177_11
def innerCandidates177_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates177_10_checked : candidates edge 4 1069 4909 innerTail177_11 best.bound = innerCandidates177_10 := by
  unfold candidates
  have row : edge 4909 = SearchData.row4909.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan177_10 : scan edge (search edge 3) 4 [1069] 1069 innerTail177_10 best = best := by
  rw [innerTail177_10, scan, if_neg (by decide), innerCandidates177_10_checked]
  change scan edge (search edge 3) 4 [1069] 1069 innerTail177_11
    (if 3 ≤ innerCandidates177_10.length then search edge 3 [1069, 4909] innerCandidates177_10 5978 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan177_11

def innerTail177_9 : List Nat := 4261 :: innerTail177_10
def innerCandidates177_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates177_9_checked : candidates edge 4 1069 4261 innerTail177_10 best.bound = innerCandidates177_9 := by
  unfold candidates
  have row : edge 4261 = SearchData.row4261.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan177_9 : scan edge (search edge 3) 4 [1069] 1069 innerTail177_9 best = best := by
  rw [innerTail177_9, scan, if_neg (by decide), innerCandidates177_9_checked]
  change scan edge (search edge 3) 4 [1069] 1069 innerTail177_10
    (if 3 ≤ innerCandidates177_9.length then search edge 3 [1069, 4261] innerCandidates177_9 5330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan177_10

def innerTail177_8 : List Nat := 4219 :: innerTail177_9
def innerCandidates177_8 : List Nat := [6337, 6529, 6883]
set_option maxRecDepth 4096 in
theorem innerCandidates177_8_checked : candidates edge 4 1069 4219 innerTail177_9 best.bound = innerCandidates177_8 := by
  unfold candidates
  have row : edge 4219 = SearchData.row4219.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild177_8 : search edge 3 [1069, 4219] innerCandidates177_8 5288 best = best := by decide

theorem innerScan177_8 : scan edge (search edge 3) 4 [1069] 1069 innerTail177_8 best = best := by
  rw [innerTail177_8, scan, if_neg (by decide), innerCandidates177_8_checked]
  change scan edge (search edge 3) 4 [1069] 1069 innerTail177_9
    (if 3 ≤ innerCandidates177_8.length then search edge 3 [1069, 4219] innerCandidates177_8 5288 best else best) = best
  rw [if_pos (by decide), innerChild177_8]
  exact innerScan177_9

def innerTail177_7 : List Nat := 3313 :: innerTail177_8
def innerCandidates177_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates177_7_checked : candidates edge 4 1069 3313 innerTail177_8 best.bound = innerCandidates177_7 := by
  unfold candidates
  have row : edge 3313 = SearchData.row3313.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan177_7 : scan edge (search edge 3) 4 [1069] 1069 innerTail177_7 best = best := by
  rw [innerTail177_7, scan, if_neg (by decide), innerCandidates177_7_checked]
  change scan edge (search edge 3) 4 [1069] 1069 innerTail177_8
    (if 3 ≤ innerCandidates177_7.length then search edge 3 [1069, 3313] innerCandidates177_7 4382 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan177_8

def innerTail177_6 : List Nat := 3061 :: innerTail177_7
def innerCandidates177_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates177_6_checked : candidates edge 4 1069 3061 innerTail177_7 best.bound = innerCandidates177_6 := by
  unfold candidates
  have row : edge 3061 = SearchData.row3061.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan177_6 : scan edge (search edge 3) 4 [1069] 1069 innerTail177_6 best = best := by
  rw [innerTail177_6, scan, if_neg (by decide), innerCandidates177_6_checked]
  change scan edge (search edge 3) 4 [1069] 1069 innerTail177_7
    (if 3 ≤ innerCandidates177_6.length then search edge 3 [1069, 3061] innerCandidates177_6 4130 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan177_7

def innerTail177_5 : List Nat := 2659 :: innerTail177_6
def innerCandidates177_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates177_5_checked : candidates edge 4 1069 2659 innerTail177_6 best.bound = innerCandidates177_5 := by
  unfold candidates
  have row : edge 2659 = PrimePairCache.smallRow2659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan177_5 : scan edge (search edge 3) 4 [1069] 1069 innerTail177_5 best = best := by
  rw [innerTail177_5, scan, if_neg (by decide), innerCandidates177_5_checked]
  change scan edge (search edge 3) 4 [1069] 1069 innerTail177_6
    (if 3 ≤ innerCandidates177_5.length then search edge 3 [1069, 2659] innerCandidates177_5 3728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan177_6

def innerTail177_4 : List Nat := 2281 :: innerTail177_5
def innerCandidates177_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates177_4_checked : candidates edge 4 1069 2281 innerTail177_5 best.bound = innerCandidates177_4 := by
  unfold candidates
  have row : edge 2281 = PrimePairCache.smallRow2281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan177_4 : scan edge (search edge 3) 4 [1069] 1069 innerTail177_4 best = best := by
  rw [innerTail177_4, scan, if_neg (by decide), innerCandidates177_4_checked]
  change scan edge (search edge 3) 4 [1069] 1069 innerTail177_5
    (if 3 ≤ innerCandidates177_4.length then search edge 3 [1069, 2281] innerCandidates177_4 3350 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan177_5

def innerTail177_3 : List Nat := 2203 :: innerTail177_4
def innerCandidates177_3 : List Nat := [3061, 6529, 8161]
set_option maxRecDepth 4096 in
theorem innerCandidates177_3_checked : candidates edge 4 1069 2203 innerTail177_4 best.bound = innerCandidates177_3 := by
  unfold candidates
  have row : edge 2203 = PrimePairCache.smallRow2203.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild177_3 : search edge 3 [1069, 2203] innerCandidates177_3 3272 best = best := by decide

theorem innerScan177_3 : scan edge (search edge 3) 4 [1069] 1069 innerTail177_3 best = best := by
  rw [innerTail177_3, scan, if_neg (by decide), innerCandidates177_3_checked]
  change scan edge (search edge 3) 4 [1069] 1069 innerTail177_4
    (if 3 ≤ innerCandidates177_3.length then search edge 3 [1069, 2203] innerCandidates177_3 3272 best else best) = best
  rw [if_pos (by decide), innerChild177_3]
  exact innerScan177_4

def innerTail177_2 : List Nat := 2113 :: innerTail177_3
def innerCandidates177_2 : List Nat := [2659]
set_option maxRecDepth 4096 in
theorem innerCandidates177_2_checked : candidates edge 4 1069 2113 innerTail177_3 best.bound = innerCandidates177_2 := by
  unfold candidates
  have row : edge 2113 = PrimePairCache.smallRow2113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan177_2 : scan edge (search edge 3) 4 [1069] 1069 innerTail177_2 best = best := by
  rw [innerTail177_2, scan, if_neg (by decide), innerCandidates177_2_checked]
  change scan edge (search edge 3) 4 [1069] 1069 innerTail177_3
    (if 3 ≤ innerCandidates177_2.length then search edge 3 [1069, 2113] innerCandidates177_2 3182 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan177_3

def innerTail177_1 : List Nat := 1609 :: innerTail177_2
def innerCandidates177_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates177_1_checked : candidates edge 4 1069 1609 innerTail177_2 best.bound = innerCandidates177_1 := by
  unfold candidates
  have row : edge 1609 = PrimePairCache.smallRow1609.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan177_1 : scan edge (search edge 3) 4 [1069] 1069 innerTail177_1 best = best := by
  rw [innerTail177_1, scan, if_neg (by decide), innerCandidates177_1_checked]
  change scan edge (search edge 3) 4 [1069] 1069 innerTail177_2
    (if 3 ≤ innerCandidates177_1.length then search edge 3 [1069, 1609] innerCandidates177_1 2678 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan177_2

def innerTail177_0 : List Nat := 1321 :: innerTail177_1
def innerCandidates177_0 : List Nat := [4261]
set_option maxRecDepth 4096 in
theorem innerCandidates177_0_checked : candidates edge 4 1069 1321 innerTail177_1 best.bound = innerCandidates177_0 := by
  unfold candidates
  have row : edge 1321 = PrimePairCache.smallRow1321.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan177_0 : scan edge (search edge 3) 4 [1069] 1069 innerTail177_0 best = best := by
  rw [innerTail177_0, scan, if_neg (by decide), innerCandidates177_0_checked]
  change scan edge (search edge 3) 4 [1069] 1069 innerTail177_1
    (if 3 ≤ innerCandidates177_0.length then search edge 3 [1069, 1321] innerCandidates177_0 2390 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan177_1

theorem rootBranch177 : search edge 4 [1069] rootCandidates177 1069 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1069] 1069 innerTail177_0 best = best
  exact innerScan177_0

end ProjectEulerSolutions.PrimePairUpper.Search
