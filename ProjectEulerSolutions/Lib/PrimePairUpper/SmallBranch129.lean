import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates287 : List Nat := [2087, 2129, 2243, 2633, 2777, 2843, 2927, 2999, 3491, 4019, 4919, 5297, 5417, 6833, 7019, 8123, 8291]
set_option maxRecDepth 4096 in
theorem rootCandidates287_checked : candidates edge 5 0 1889 tail288 best.bound = rootCandidates287 := by
  unfold candidates
  have row : edge 1889 = PrimePairCache.smallRow1889.contains := by funext q; rfl
  rw [row]
  decide

def innerTail287_17 : List Nat := []
theorem innerScan287_17 : scan edge (search edge 3) 4 [1889] 1889 innerTail287_17 best = best := rfl

def innerTail287_16 : List Nat := 8291 :: innerTail287_17
def innerCandidates287_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates287_16_checked : candidates edge 4 1889 8291 innerTail287_17 best.bound = innerCandidates287_16 := by
  unfold candidates
  have row : edge 8291 = SearchData.row8291.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan287_16 : scan edge (search edge 3) 4 [1889] 1889 innerTail287_16 best = best := by
  rw [innerTail287_16, scan, if_neg (by decide), innerCandidates287_16_checked]
  change scan edge (search edge 3) 4 [1889] 1889 innerTail287_17
    (if 3 ≤ innerCandidates287_16.length then search edge 3 [1889, 8291] innerCandidates287_16 10180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan287_17

def innerTail287_15 : List Nat := 8123 :: innerTail287_16
def innerCandidates287_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates287_15_checked : candidates edge 4 1889 8123 innerTail287_16 best.bound = innerCandidates287_15 := by
  unfold candidates
  have row : edge 8123 = SearchData.row8123.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan287_15 : scan edge (search edge 3) 4 [1889] 1889 innerTail287_15 best = best := by
  rw [innerTail287_15, scan, if_neg (by decide), innerCandidates287_15_checked]
  change scan edge (search edge 3) 4 [1889] 1889 innerTail287_16
    (if 3 ≤ innerCandidates287_15.length then search edge 3 [1889, 8123] innerCandidates287_15 10012 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan287_16

def innerTail287_14 : List Nat := 7019 :: innerTail287_15
def innerCandidates287_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates287_14_checked : candidates edge 4 1889 7019 innerTail287_15 best.bound = innerCandidates287_14 := by
  unfold candidates
  have row : edge 7019 = SearchData.row7019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan287_14 : scan edge (search edge 3) 4 [1889] 1889 innerTail287_14 best = best := by
  rw [innerTail287_14, scan, if_neg (by decide), innerCandidates287_14_checked]
  change scan edge (search edge 3) 4 [1889] 1889 innerTail287_15
    (if 3 ≤ innerCandidates287_14.length then search edge 3 [1889, 7019] innerCandidates287_14 8908 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan287_15

def innerTail287_13 : List Nat := 6833 :: innerTail287_14
def innerCandidates287_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates287_13_checked : candidates edge 4 1889 6833 innerTail287_14 best.bound = innerCandidates287_13 := by
  unfold candidates
  have row : edge 6833 = SearchData.row6833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan287_13 : scan edge (search edge 3) 4 [1889] 1889 innerTail287_13 best = best := by
  rw [innerTail287_13, scan, if_neg (by decide), innerCandidates287_13_checked]
  change scan edge (search edge 3) 4 [1889] 1889 innerTail287_14
    (if 3 ≤ innerCandidates287_13.length then search edge 3 [1889, 6833] innerCandidates287_13 8722 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan287_14

def innerTail287_12 : List Nat := 5417 :: innerTail287_13
def innerCandidates287_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates287_12_checked : candidates edge 4 1889 5417 innerTail287_13 best.bound = innerCandidates287_12 := by
  unfold candidates
  have row : edge 5417 = SearchData.row5417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan287_12 : scan edge (search edge 3) 4 [1889] 1889 innerTail287_12 best = best := by
  rw [innerTail287_12, scan, if_neg (by decide), innerCandidates287_12_checked]
  change scan edge (search edge 3) 4 [1889] 1889 innerTail287_13
    (if 3 ≤ innerCandidates287_12.length then search edge 3 [1889, 5417] innerCandidates287_12 7306 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan287_13

def innerTail287_11 : List Nat := 5297 :: innerTail287_12
def innerCandidates287_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates287_11_checked : candidates edge 4 1889 5297 innerTail287_12 best.bound = innerCandidates287_11 := by
  unfold candidates
  have row : edge 5297 = SearchData.row5297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan287_11 : scan edge (search edge 3) 4 [1889] 1889 innerTail287_11 best = best := by
  rw [innerTail287_11, scan, if_neg (by decide), innerCandidates287_11_checked]
  change scan edge (search edge 3) 4 [1889] 1889 innerTail287_12
    (if 3 ≤ innerCandidates287_11.length then search edge 3 [1889, 5297] innerCandidates287_11 7186 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan287_12

def innerTail287_10 : List Nat := 4919 :: innerTail287_11
def innerCandidates287_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates287_10_checked : candidates edge 4 1889 4919 innerTail287_11 best.bound = innerCandidates287_10 := by
  unfold candidates
  have row : edge 4919 = SearchData.row4919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan287_10 : scan edge (search edge 3) 4 [1889] 1889 innerTail287_10 best = best := by
  rw [innerTail287_10, scan, if_neg (by decide), innerCandidates287_10_checked]
  change scan edge (search edge 3) 4 [1889] 1889 innerTail287_11
    (if 3 ≤ innerCandidates287_10.length then search edge 3 [1889, 4919] innerCandidates287_10 6808 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan287_11

def innerTail287_9 : List Nat := 4019 :: innerTail287_10
def innerCandidates287_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates287_9_checked : candidates edge 4 1889 4019 innerTail287_10 best.bound = innerCandidates287_9 := by
  unfold candidates
  have row : edge 4019 = SearchData.row4019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan287_9 : scan edge (search edge 3) 4 [1889] 1889 innerTail287_9 best = best := by
  rw [innerTail287_9, scan, if_neg (by decide), innerCandidates287_9_checked]
  change scan edge (search edge 3) 4 [1889] 1889 innerTail287_10
    (if 3 ≤ innerCandidates287_9.length then search edge 3 [1889, 4019] innerCandidates287_9 5908 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan287_10

def innerTail287_8 : List Nat := 3491 :: innerTail287_9
def innerCandidates287_8 : List Nat := [8123]
set_option maxRecDepth 4096 in
theorem innerCandidates287_8_checked : candidates edge 4 1889 3491 innerTail287_9 best.bound = innerCandidates287_8 := by
  unfold candidates
  have row : edge 3491 = SearchData.row3491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan287_8 : scan edge (search edge 3) 4 [1889] 1889 innerTail287_8 best = best := by
  rw [innerTail287_8, scan, if_neg (by decide), innerCandidates287_8_checked]
  change scan edge (search edge 3) 4 [1889] 1889 innerTail287_9
    (if 3 ≤ innerCandidates287_8.length then search edge 3 [1889, 3491] innerCandidates287_8 5380 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan287_9

def innerTail287_7 : List Nat := 2999 :: innerTail287_8
def innerCandidates287_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates287_7_checked : candidates edge 4 1889 2999 innerTail287_8 best.bound = innerCandidates287_7 := by
  unfold candidates
  have row : edge 2999 = SearchData.row2999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan287_7 : scan edge (search edge 3) 4 [1889] 1889 innerTail287_7 best = best := by
  rw [innerTail287_7, scan, if_neg (by decide), innerCandidates287_7_checked]
  change scan edge (search edge 3) 4 [1889] 1889 innerTail287_8
    (if 3 ≤ innerCandidates287_7.length then search edge 3 [1889, 2999] innerCandidates287_7 4888 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan287_8

def innerTail287_6 : List Nat := 2927 :: innerTail287_7
def innerCandidates287_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates287_6_checked : candidates edge 4 1889 2927 innerTail287_7 best.bound = innerCandidates287_6 := by
  unfold candidates
  have row : edge 2927 = SearchData.row2927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan287_6 : scan edge (search edge 3) 4 [1889] 1889 innerTail287_6 best = best := by
  rw [innerTail287_6, scan, if_neg (by decide), innerCandidates287_6_checked]
  change scan edge (search edge 3) 4 [1889] 1889 innerTail287_7
    (if 3 ≤ innerCandidates287_6.length then search edge 3 [1889, 2927] innerCandidates287_6 4816 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan287_7

def innerTail287_5 : List Nat := 2843 :: innerTail287_6
def innerCandidates287_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates287_5_checked : candidates edge 4 1889 2843 innerTail287_6 best.bound = innerCandidates287_5 := by
  unfold candidates
  have row : edge 2843 = SearchData.row2843.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan287_5 : scan edge (search edge 3) 4 [1889] 1889 innerTail287_5 best = best := by
  rw [innerTail287_5, scan, if_neg (by decide), innerCandidates287_5_checked]
  change scan edge (search edge 3) 4 [1889] 1889 innerTail287_6
    (if 3 ≤ innerCandidates287_5.length then search edge 3 [1889, 2843] innerCandidates287_5 4732 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan287_6

def innerTail287_4 : List Nat := 2777 :: innerTail287_5
def innerCandidates287_4 : List Nat := [8291]
set_option maxRecDepth 4096 in
theorem innerCandidates287_4_checked : candidates edge 4 1889 2777 innerTail287_5 best.bound = innerCandidates287_4 := by
  unfold candidates
  have row : edge 2777 = SearchData.row2777.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan287_4 : scan edge (search edge 3) 4 [1889] 1889 innerTail287_4 best = best := by
  rw [innerTail287_4, scan, if_neg (by decide), innerCandidates287_4_checked]
  change scan edge (search edge 3) 4 [1889] 1889 innerTail287_5
    (if 3 ≤ innerCandidates287_4.length then search edge 3 [1889, 2777] innerCandidates287_4 4666 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan287_5

def innerTail287_3 : List Nat := 2633 :: innerTail287_4
def innerCandidates287_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates287_3_checked : candidates edge 4 1889 2633 innerTail287_4 best.bound = innerCandidates287_3 := by
  unfold candidates
  have row : edge 2633 = PrimePairCache.smallRow2633.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan287_3 : scan edge (search edge 3) 4 [1889] 1889 innerTail287_3 best = best := by
  rw [innerTail287_3, scan, if_neg (by decide), innerCandidates287_3_checked]
  change scan edge (search edge 3) 4 [1889] 1889 innerTail287_4
    (if 3 ≤ innerCandidates287_3.length then search edge 3 [1889, 2633] innerCandidates287_3 4522 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan287_4

def innerTail287_2 : List Nat := 2243 :: innerTail287_3
def innerCandidates287_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates287_2_checked : candidates edge 4 1889 2243 innerTail287_3 best.bound = innerCandidates287_2 := by
  unfold candidates
  have row : edge 2243 = PrimePairCache.smallRow2243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan287_2 : scan edge (search edge 3) 4 [1889] 1889 innerTail287_2 best = best := by
  rw [innerTail287_2, scan, if_neg (by decide), innerCandidates287_2_checked]
  change scan edge (search edge 3) 4 [1889] 1889 innerTail287_3
    (if 3 ≤ innerCandidates287_2.length then search edge 3 [1889, 2243] innerCandidates287_2 4132 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan287_3

def innerTail287_1 : List Nat := 2129 :: innerTail287_2
def innerCandidates287_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates287_1_checked : candidates edge 4 1889 2129 innerTail287_2 best.bound = innerCandidates287_1 := by
  unfold candidates
  have row : edge 2129 = PrimePairCache.smallRow2129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan287_1 : scan edge (search edge 3) 4 [1889] 1889 innerTail287_1 best = best := by
  rw [innerTail287_1, scan, if_neg (by decide), innerCandidates287_1_checked]
  change scan edge (search edge 3) 4 [1889] 1889 innerTail287_2
    (if 3 ≤ innerCandidates287_1.length then search edge 3 [1889, 2129] innerCandidates287_1 4018 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan287_2

def innerTail287_0 : List Nat := 2087 :: innerTail287_1
def innerCandidates287_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates287_0_checked : candidates edge 4 1889 2087 innerTail287_1 best.bound = innerCandidates287_0 := by
  unfold candidates
  have row : edge 2087 = PrimePairCache.smallRow2087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan287_0 : scan edge (search edge 3) 4 [1889] 1889 innerTail287_0 best = best := by
  rw [innerTail287_0, scan, if_neg (by decide), innerCandidates287_0_checked]
  change scan edge (search edge 3) 4 [1889] 1889 innerTail287_1
    (if 3 ≤ innerCandidates287_0.length then search edge 3 [1889, 2087] innerCandidates287_0 3976 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan287_1

theorem rootBranch287 : search edge 4 [1889] rootCandidates287 1889 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1889] 1889 innerTail287_0 best = best
  exact innerScan287_0

def rootCandidates286 : List Nat := [1999, 2281, 2341, 2713, 2971, 3331, 4861, 6823, 7297, 7411, 7951]
set_option maxRecDepth 4096 in
theorem rootCandidates286_checked : candidates edge 5 0 1879 tail287 best.bound = rootCandidates286 := by
  unfold candidates
  have row : edge 1879 = PrimePairCache.smallRow1879.contains := by funext q; rfl
  rw [row]
  decide

def innerTail286_11 : List Nat := []
theorem innerScan286_11 : scan edge (search edge 3) 4 [1879] 1879 innerTail286_11 best = best := rfl

def innerTail286_10 : List Nat := 7951 :: innerTail286_11
def innerCandidates286_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates286_10_checked : candidates edge 4 1879 7951 innerTail286_11 best.bound = innerCandidates286_10 := by
  unfold candidates
  have row : edge 7951 = SearchData.row7951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan286_10 : scan edge (search edge 3) 4 [1879] 1879 innerTail286_10 best = best := by
  rw [innerTail286_10, scan, if_neg (by decide), innerCandidates286_10_checked]
  change scan edge (search edge 3) 4 [1879] 1879 innerTail286_11
    (if 3 ≤ innerCandidates286_10.length then search edge 3 [1879, 7951] innerCandidates286_10 9830 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan286_11

def innerTail286_9 : List Nat := 7411 :: innerTail286_10
def innerCandidates286_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates286_9_checked : candidates edge 4 1879 7411 innerTail286_10 best.bound = innerCandidates286_9 := by
  unfold candidates
  have row : edge 7411 = SearchData.row7411.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan286_9 : scan edge (search edge 3) 4 [1879] 1879 innerTail286_9 best = best := by
  rw [innerTail286_9, scan, if_neg (by decide), innerCandidates286_9_checked]
  change scan edge (search edge 3) 4 [1879] 1879 innerTail286_10
    (if 3 ≤ innerCandidates286_9.length then search edge 3 [1879, 7411] innerCandidates286_9 9290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan286_10

def innerTail286_8 : List Nat := 7297 :: innerTail286_9
def innerCandidates286_8 : List Nat := [7951]
set_option maxRecDepth 4096 in
theorem innerCandidates286_8_checked : candidates edge 4 1879 7297 innerTail286_9 best.bound = innerCandidates286_8 := by
  unfold candidates
  have row : edge 7297 = SearchData.row7297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan286_8 : scan edge (search edge 3) 4 [1879] 1879 innerTail286_8 best = best := by
  rw [innerTail286_8, scan, if_neg (by decide), innerCandidates286_8_checked]
  change scan edge (search edge 3) 4 [1879] 1879 innerTail286_9
    (if 3 ≤ innerCandidates286_8.length then search edge 3 [1879, 7297] innerCandidates286_8 9176 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan286_9

def innerTail286_7 : List Nat := 6823 :: innerTail286_8
def innerCandidates286_7 : List Nat := [7951]
set_option maxRecDepth 4096 in
theorem innerCandidates286_7_checked : candidates edge 4 1879 6823 innerTail286_8 best.bound = innerCandidates286_7 := by
  unfold candidates
  have row : edge 6823 = SearchData.row6823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan286_7 : scan edge (search edge 3) 4 [1879] 1879 innerTail286_7 best = best := by
  rw [innerTail286_7, scan, if_neg (by decide), innerCandidates286_7_checked]
  change scan edge (search edge 3) 4 [1879] 1879 innerTail286_8
    (if 3 ≤ innerCandidates286_7.length then search edge 3 [1879, 6823] innerCandidates286_7 8702 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan286_8

def innerTail286_6 : List Nat := 4861 :: innerTail286_7
def innerCandidates286_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates286_6_checked : candidates edge 4 1879 4861 innerTail286_7 best.bound = innerCandidates286_6 := by
  unfold candidates
  have row : edge 4861 = SearchData.row4861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan286_6 : scan edge (search edge 3) 4 [1879] 1879 innerTail286_6 best = best := by
  rw [innerTail286_6, scan, if_neg (by decide), innerCandidates286_6_checked]
  change scan edge (search edge 3) 4 [1879] 1879 innerTail286_7
    (if 3 ≤ innerCandidates286_6.length then search edge 3 [1879, 4861] innerCandidates286_6 6740 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan286_7

def innerTail286_5 : List Nat := 3331 :: innerTail286_6
def innerCandidates286_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates286_5_checked : candidates edge 4 1879 3331 innerTail286_6 best.bound = innerCandidates286_5 := by
  unfold candidates
  have row : edge 3331 = SearchData.row3331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan286_5 : scan edge (search edge 3) 4 [1879] 1879 innerTail286_5 best = best := by
  rw [innerTail286_5, scan, if_neg (by decide), innerCandidates286_5_checked]
  change scan edge (search edge 3) 4 [1879] 1879 innerTail286_6
    (if 3 ≤ innerCandidates286_5.length then search edge 3 [1879, 3331] innerCandidates286_5 5210 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan286_6

def innerTail286_4 : List Nat := 2971 :: innerTail286_5
def innerCandidates286_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates286_4_checked : candidates edge 4 1879 2971 innerTail286_5 best.bound = innerCandidates286_4 := by
  unfold candidates
  have row : edge 2971 = SearchData.row2971.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan286_4 : scan edge (search edge 3) 4 [1879] 1879 innerTail286_4 best = best := by
  rw [innerTail286_4, scan, if_neg (by decide), innerCandidates286_4_checked]
  change scan edge (search edge 3) 4 [1879] 1879 innerTail286_5
    (if 3 ≤ innerCandidates286_4.length then search edge 3 [1879, 2971] innerCandidates286_4 4850 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan286_5

def innerTail286_3 : List Nat := 2713 :: innerTail286_4
def innerCandidates286_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates286_3_checked : candidates edge 4 1879 2713 innerTail286_4 best.bound = innerCandidates286_3 := by
  unfold candidates
  have row : edge 2713 = SearchData.row2713.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan286_3 : scan edge (search edge 3) 4 [1879] 1879 innerTail286_3 best = best := by
  rw [innerTail286_3, scan, if_neg (by decide), innerCandidates286_3_checked]
  change scan edge (search edge 3) 4 [1879] 1879 innerTail286_4
    (if 3 ≤ innerCandidates286_3.length then search edge 3 [1879, 2713] innerCandidates286_3 4592 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan286_4

def innerTail286_2 : List Nat := 2341 :: innerTail286_3
def innerCandidates286_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates286_2_checked : candidates edge 4 1879 2341 innerTail286_3 best.bound = innerCandidates286_2 := by
  unfold candidates
  have row : edge 2341 = PrimePairCache.smallRow2341.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan286_2 : scan edge (search edge 3) 4 [1879] 1879 innerTail286_2 best = best := by
  rw [innerTail286_2, scan, if_neg (by decide), innerCandidates286_2_checked]
  change scan edge (search edge 3) 4 [1879] 1879 innerTail286_3
    (if 3 ≤ innerCandidates286_2.length then search edge 3 [1879, 2341] innerCandidates286_2 4220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan286_3

def innerTail286_1 : List Nat := 2281 :: innerTail286_2
def innerCandidates286_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates286_1_checked : candidates edge 4 1879 2281 innerTail286_2 best.bound = innerCandidates286_1 := by
  unfold candidates
  have row : edge 2281 = PrimePairCache.smallRow2281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan286_1 : scan edge (search edge 3) 4 [1879] 1879 innerTail286_1 best = best := by
  rw [innerTail286_1, scan, if_neg (by decide), innerCandidates286_1_checked]
  change scan edge (search edge 3) 4 [1879] 1879 innerTail286_2
    (if 3 ≤ innerCandidates286_1.length then search edge 3 [1879, 2281] innerCandidates286_1 4160 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan286_2

def innerTail286_0 : List Nat := 1999 :: innerTail286_1
def innerCandidates286_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates286_0_checked : candidates edge 4 1879 1999 innerTail286_1 best.bound = innerCandidates286_0 := by
  unfold candidates
  have row : edge 1999 = PrimePairCache.smallRow1999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan286_0 : scan edge (search edge 3) 4 [1879] 1879 innerTail286_0 best = best := by
  rw [innerTail286_0, scan, if_neg (by decide), innerCandidates286_0_checked]
  change scan edge (search edge 3) 4 [1879] 1879 innerTail286_1
    (if 3 ≤ innerCandidates286_0.length then search edge 3 [1879, 1999] innerCandidates286_0 3878 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan286_1

theorem rootBranch286 : search edge 4 [1879] rootCandidates286 1879 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1879] 1879 innerTail286_0 best = best
  exact innerScan286_0

def rootCandidates285 : List Nat := [1889, 2447, 2549, 2909, 3119, 3917, 4211, 4391, 4421, 4733, 5261, 5399, 5711, 6113, 6287, 6491, 6971, 6983, 7823, 8087]
set_option maxRecDepth 4096 in
theorem rootCandidates285_checked : candidates edge 5 0 1877 tail286 best.bound = rootCandidates285 := by
  unfold candidates
  have row : edge 1877 = PrimePairCache.smallRow1877.contains := by funext q; rfl
  rw [row]
  decide

def innerTail285_20 : List Nat := []
theorem innerScan285_20 : scan edge (search edge 3) 4 [1877] 1877 innerTail285_20 best = best := rfl

def innerTail285_19 : List Nat := 8087 :: innerTail285_20
def innerCandidates285_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates285_19_checked : candidates edge 4 1877 8087 innerTail285_20 best.bound = innerCandidates285_19 := by
  unfold candidates
  have row : edge 8087 = SearchData.row8087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan285_19 : scan edge (search edge 3) 4 [1877] 1877 innerTail285_19 best = best := by
  rw [innerTail285_19, scan, if_neg (by decide), innerCandidates285_19_checked]
  change scan edge (search edge 3) 4 [1877] 1877 innerTail285_20
    (if 3 ≤ innerCandidates285_19.length then search edge 3 [1877, 8087] innerCandidates285_19 9964 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan285_20

def innerTail285_18 : List Nat := 7823 :: innerTail285_19
def innerCandidates285_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates285_18_checked : candidates edge 4 1877 7823 innerTail285_19 best.bound = innerCandidates285_18 := by
  unfold candidates
  have row : edge 7823 = SearchData.row7823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan285_18 : scan edge (search edge 3) 4 [1877] 1877 innerTail285_18 best = best := by
  rw [innerTail285_18, scan, if_neg (by decide), innerCandidates285_18_checked]
  change scan edge (search edge 3) 4 [1877] 1877 innerTail285_19
    (if 3 ≤ innerCandidates285_18.length then search edge 3 [1877, 7823] innerCandidates285_18 9700 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan285_19

def innerTail285_17 : List Nat := 6983 :: innerTail285_18
def innerCandidates285_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates285_17_checked : candidates edge 4 1877 6983 innerTail285_18 best.bound = innerCandidates285_17 := by
  unfold candidates
  have row : edge 6983 = SearchData.row6983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan285_17 : scan edge (search edge 3) 4 [1877] 1877 innerTail285_17 best = best := by
  rw [innerTail285_17, scan, if_neg (by decide), innerCandidates285_17_checked]
  change scan edge (search edge 3) 4 [1877] 1877 innerTail285_18
    (if 3 ≤ innerCandidates285_17.length then search edge 3 [1877, 6983] innerCandidates285_17 8860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan285_18

def innerTail285_16 : List Nat := 6971 :: innerTail285_17
def innerCandidates285_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates285_16_checked : candidates edge 4 1877 6971 innerTail285_17 best.bound = innerCandidates285_16 := by
  unfold candidates
  have row : edge 6971 = SearchData.row6971.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan285_16 : scan edge (search edge 3) 4 [1877] 1877 innerTail285_16 best = best := by
  rw [innerTail285_16, scan, if_neg (by decide), innerCandidates285_16_checked]
  change scan edge (search edge 3) 4 [1877] 1877 innerTail285_17
    (if 3 ≤ innerCandidates285_16.length then search edge 3 [1877, 6971] innerCandidates285_16 8848 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan285_17

def innerTail285_15 : List Nat := 6491 :: innerTail285_16
def innerCandidates285_15 : List Nat := [6983]
set_option maxRecDepth 4096 in
theorem innerCandidates285_15_checked : candidates edge 4 1877 6491 innerTail285_16 best.bound = innerCandidates285_15 := by
  unfold candidates
  have row : edge 6491 = SearchData.row6491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan285_15 : scan edge (search edge 3) 4 [1877] 1877 innerTail285_15 best = best := by
  rw [innerTail285_15, scan, if_neg (by decide), innerCandidates285_15_checked]
  change scan edge (search edge 3) 4 [1877] 1877 innerTail285_16
    (if 3 ≤ innerCandidates285_15.length then search edge 3 [1877, 6491] innerCandidates285_15 8368 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan285_16

def innerTail285_14 : List Nat := 6287 :: innerTail285_15
def innerCandidates285_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates285_14_checked : candidates edge 4 1877 6287 innerTail285_15 best.bound = innerCandidates285_14 := by
  unfold candidates
  have row : edge 6287 = SearchData.row6287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan285_14 : scan edge (search edge 3) 4 [1877] 1877 innerTail285_14 best = best := by
  rw [innerTail285_14, scan, if_neg (by decide), innerCandidates285_14_checked]
  change scan edge (search edge 3) 4 [1877] 1877 innerTail285_15
    (if 3 ≤ innerCandidates285_14.length then search edge 3 [1877, 6287] innerCandidates285_14 8164 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan285_15

def innerTail285_13 : List Nat := 6113 :: innerTail285_14
def innerCandidates285_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates285_13_checked : candidates edge 4 1877 6113 innerTail285_14 best.bound = innerCandidates285_13 := by
  unfold candidates
  have row : edge 6113 = SearchData.row6113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan285_13 : scan edge (search edge 3) 4 [1877] 1877 innerTail285_13 best = best := by
  rw [innerTail285_13, scan, if_neg (by decide), innerCandidates285_13_checked]
  change scan edge (search edge 3) 4 [1877] 1877 innerTail285_14
    (if 3 ≤ innerCandidates285_13.length then search edge 3 [1877, 6113] innerCandidates285_13 7990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan285_14

def innerTail285_12 : List Nat := 5711 :: innerTail285_13
def innerCandidates285_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates285_12_checked : candidates edge 4 1877 5711 innerTail285_13 best.bound = innerCandidates285_12 := by
  unfold candidates
  have row : edge 5711 = SearchData.row5711.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan285_12 : scan edge (search edge 3) 4 [1877] 1877 innerTail285_12 best = best := by
  rw [innerTail285_12, scan, if_neg (by decide), innerCandidates285_12_checked]
  change scan edge (search edge 3) 4 [1877] 1877 innerTail285_13
    (if 3 ≤ innerCandidates285_12.length then search edge 3 [1877, 5711] innerCandidates285_12 7588 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan285_13

def innerTail285_11 : List Nat := 5399 :: innerTail285_12
def innerCandidates285_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates285_11_checked : candidates edge 4 1877 5399 innerTail285_12 best.bound = innerCandidates285_11 := by
  unfold candidates
  have row : edge 5399 = SearchData.row5399.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan285_11 : scan edge (search edge 3) 4 [1877] 1877 innerTail285_11 best = best := by
  rw [innerTail285_11, scan, if_neg (by decide), innerCandidates285_11_checked]
  change scan edge (search edge 3) 4 [1877] 1877 innerTail285_12
    (if 3 ≤ innerCandidates285_11.length then search edge 3 [1877, 5399] innerCandidates285_11 7276 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan285_12

def innerTail285_10 : List Nat := 5261 :: innerTail285_11
def innerCandidates285_10 : List Nat := [6983]
set_option maxRecDepth 4096 in
theorem innerCandidates285_10_checked : candidates edge 4 1877 5261 innerTail285_11 best.bound = innerCandidates285_10 := by
  unfold candidates
  have row : edge 5261 = SearchData.row5261.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan285_10 : scan edge (search edge 3) 4 [1877] 1877 innerTail285_10 best = best := by
  rw [innerTail285_10, scan, if_neg (by decide), innerCandidates285_10_checked]
  change scan edge (search edge 3) 4 [1877] 1877 innerTail285_11
    (if 3 ≤ innerCandidates285_10.length then search edge 3 [1877, 5261] innerCandidates285_10 7138 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan285_11

def innerTail285_9 : List Nat := 4733 :: innerTail285_10
def innerCandidates285_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates285_9_checked : candidates edge 4 1877 4733 innerTail285_10 best.bound = innerCandidates285_9 := by
  unfold candidates
  have row : edge 4733 = SearchData.row4733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan285_9 : scan edge (search edge 3) 4 [1877] 1877 innerTail285_9 best = best := by
  rw [innerTail285_9, scan, if_neg (by decide), innerCandidates285_9_checked]
  change scan edge (search edge 3) 4 [1877] 1877 innerTail285_10
    (if 3 ≤ innerCandidates285_9.length then search edge 3 [1877, 4733] innerCandidates285_9 6610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan285_10

def innerTail285_8 : List Nat := 4421 :: innerTail285_9
def innerCandidates285_8 : List Nat := [6113]
set_option maxRecDepth 4096 in
theorem innerCandidates285_8_checked : candidates edge 4 1877 4421 innerTail285_9 best.bound = innerCandidates285_8 := by
  unfold candidates
  have row : edge 4421 = SearchData.row4421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan285_8 : scan edge (search edge 3) 4 [1877] 1877 innerTail285_8 best = best := by
  rw [innerTail285_8, scan, if_neg (by decide), innerCandidates285_8_checked]
  change scan edge (search edge 3) 4 [1877] 1877 innerTail285_9
    (if 3 ≤ innerCandidates285_8.length then search edge 3 [1877, 4421] innerCandidates285_8 6298 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan285_9

def innerTail285_7 : List Nat := 4391 :: innerTail285_8
def innerCandidates285_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates285_7_checked : candidates edge 4 1877 4391 innerTail285_8 best.bound = innerCandidates285_7 := by
  unfold candidates
  have row : edge 4391 = SearchData.row4391.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan285_7 : scan edge (search edge 3) 4 [1877] 1877 innerTail285_7 best = best := by
  rw [innerTail285_7, scan, if_neg (by decide), innerCandidates285_7_checked]
  change scan edge (search edge 3) 4 [1877] 1877 innerTail285_8
    (if 3 ≤ innerCandidates285_7.length then search edge 3 [1877, 4391] innerCandidates285_7 6268 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan285_8

def innerTail285_6 : List Nat := 4211 :: innerTail285_7
def innerCandidates285_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates285_6_checked : candidates edge 4 1877 4211 innerTail285_7 best.bound = innerCandidates285_6 := by
  unfold candidates
  have row : edge 4211 = SearchData.row4211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan285_6 : scan edge (search edge 3) 4 [1877] 1877 innerTail285_6 best = best := by
  rw [innerTail285_6, scan, if_neg (by decide), innerCandidates285_6_checked]
  change scan edge (search edge 3) 4 [1877] 1877 innerTail285_7
    (if 3 ≤ innerCandidates285_6.length then search edge 3 [1877, 4211] innerCandidates285_6 6088 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan285_7

def innerTail285_5 : List Nat := 3917 :: innerTail285_6
def innerCandidates285_5 : List Nat := [4211, 5399]
set_option maxRecDepth 4096 in
theorem innerCandidates285_5_checked : candidates edge 4 1877 3917 innerTail285_6 best.bound = innerCandidates285_5 := by
  unfold candidates
  have row : edge 3917 = SearchData.row3917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan285_5 : scan edge (search edge 3) 4 [1877] 1877 innerTail285_5 best = best := by
  rw [innerTail285_5, scan, if_neg (by decide), innerCandidates285_5_checked]
  change scan edge (search edge 3) 4 [1877] 1877 innerTail285_6
    (if 3 ≤ innerCandidates285_5.length then search edge 3 [1877, 3917] innerCandidates285_5 5794 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan285_6

def innerTail285_4 : List Nat := 3119 :: innerTail285_5
def innerCandidates285_4 : List Nat := [4391, 6491]
set_option maxRecDepth 4096 in
theorem innerCandidates285_4_checked : candidates edge 4 1877 3119 innerTail285_5 best.bound = innerCandidates285_4 := by
  unfold candidates
  have row : edge 3119 = SearchData.row3119.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan285_4 : scan edge (search edge 3) 4 [1877] 1877 innerTail285_4 best = best := by
  rw [innerTail285_4, scan, if_neg (by decide), innerCandidates285_4_checked]
  change scan edge (search edge 3) 4 [1877] 1877 innerTail285_5
    (if 3 ≤ innerCandidates285_4.length then search edge 3 [1877, 3119] innerCandidates285_4 4996 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan285_5

def innerTail285_3 : List Nat := 2909 :: innerTail285_4
def innerCandidates285_3 : List Nat := [6983]
set_option maxRecDepth 4096 in
theorem innerCandidates285_3_checked : candidates edge 4 1877 2909 innerTail285_4 best.bound = innerCandidates285_3 := by
  unfold candidates
  have row : edge 2909 = SearchData.row2909.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan285_3 : scan edge (search edge 3) 4 [1877] 1877 innerTail285_3 best = best := by
  rw [innerTail285_3, scan, if_neg (by decide), innerCandidates285_3_checked]
  change scan edge (search edge 3) 4 [1877] 1877 innerTail285_4
    (if 3 ≤ innerCandidates285_3.length then search edge 3 [1877, 2909] innerCandidates285_3 4786 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan285_4

def innerTail285_2 : List Nat := 2549 :: innerTail285_3
def innerCandidates285_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates285_2_checked : candidates edge 4 1877 2549 innerTail285_3 best.bound = innerCandidates285_2 := by
  unfold candidates
  have row : edge 2549 = PrimePairCache.smallRow2549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan285_2 : scan edge (search edge 3) 4 [1877] 1877 innerTail285_2 best = best := by
  rw [innerTail285_2, scan, if_neg (by decide), innerCandidates285_2_checked]
  change scan edge (search edge 3) 4 [1877] 1877 innerTail285_3
    (if 3 ≤ innerCandidates285_2.length then search edge 3 [1877, 2549] innerCandidates285_2 4426 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan285_3

def innerTail285_1 : List Nat := 2447 :: innerTail285_2
def innerCandidates285_1 : List Nat := [6113]
set_option maxRecDepth 4096 in
theorem innerCandidates285_1_checked : candidates edge 4 1877 2447 innerTail285_2 best.bound = innerCandidates285_1 := by
  unfold candidates
  have row : edge 2447 = PrimePairCache.smallRow2447.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan285_1 : scan edge (search edge 3) 4 [1877] 1877 innerTail285_1 best = best := by
  rw [innerTail285_1, scan, if_neg (by decide), innerCandidates285_1_checked]
  change scan edge (search edge 3) 4 [1877] 1877 innerTail285_2
    (if 3 ≤ innerCandidates285_1.length then search edge 3 [1877, 2447] innerCandidates285_1 4324 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan285_2

def innerTail285_0 : List Nat := 1889 :: innerTail285_1
def innerCandidates285_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates285_0_checked : candidates edge 4 1877 1889 innerTail285_1 best.bound = innerCandidates285_0 := by
  unfold candidates
  have row : edge 1889 = PrimePairCache.smallRow1889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan285_0 : scan edge (search edge 3) 4 [1877] 1877 innerTail285_0 best = best := by
  rw [innerTail285_0, scan, if_neg (by decide), innerCandidates285_0_checked]
  change scan edge (search edge 3) 4 [1877] 1877 innerTail285_1
    (if 3 ≤ innerCandidates285_0.length then search edge 3 [1877, 1889] innerCandidates285_0 3766 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan285_1

theorem rootBranch285 : search edge 4 [1877] rootCandidates285 1877 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1877] 1877 innerTail285_0 best = best
  exact innerScan285_0

end ProjectEulerSolutions.PrimePairUpper.Search
