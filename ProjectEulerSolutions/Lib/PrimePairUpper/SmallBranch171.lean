import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates169 : List Nat := [1699, 2539, 2749, 3529, 3559, 3571, 3769, 4597, 5281, 5701, 6079, 6247, 6373, 6397, 6691, 7213, 7873]
set_option maxRecDepth 4096 in
theorem rootCandidates169_checked : candidates edge 5 0 1021 tail170 best.bound = rootCandidates169 := by
  unfold candidates
  have row : edge 1021 = PrimePairCache.smallRow1021.contains := by funext q; rfl
  rw [row]
  decide

def innerTail169_17 : List Nat := []
theorem innerScan169_17 : scan edge (search edge 3) 4 [1021] 1021 innerTail169_17 best = best := rfl

def innerTail169_16 : List Nat := 7873 :: innerTail169_17
def innerCandidates169_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates169_16_checked : candidates edge 4 1021 7873 innerTail169_17 best.bound = innerCandidates169_16 := by
  unfold candidates
  have row : edge 7873 = SearchData.row7873.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan169_16 : scan edge (search edge 3) 4 [1021] 1021 innerTail169_16 best = best := by
  rw [innerTail169_16, scan, if_neg (by decide), innerCandidates169_16_checked]
  change scan edge (search edge 3) 4 [1021] 1021 innerTail169_17
    (if 3 ≤ innerCandidates169_16.length then search edge 3 [1021, 7873] innerCandidates169_16 8894 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan169_17

def innerTail169_15 : List Nat := 7213 :: innerTail169_16
def innerCandidates169_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates169_15_checked : candidates edge 4 1021 7213 innerTail169_16 best.bound = innerCandidates169_15 := by
  unfold candidates
  have row : edge 7213 = SearchData.row7213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan169_15 : scan edge (search edge 3) 4 [1021] 1021 innerTail169_15 best = best := by
  rw [innerTail169_15, scan, if_neg (by decide), innerCandidates169_15_checked]
  change scan edge (search edge 3) 4 [1021] 1021 innerTail169_16
    (if 3 ≤ innerCandidates169_15.length then search edge 3 [1021, 7213] innerCandidates169_15 8234 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan169_16

def innerTail169_14 : List Nat := 6691 :: innerTail169_15
def innerCandidates169_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates169_14_checked : candidates edge 4 1021 6691 innerTail169_15 best.bound = innerCandidates169_14 := by
  unfold candidates
  have row : edge 6691 = SearchData.row6691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan169_14 : scan edge (search edge 3) 4 [1021] 1021 innerTail169_14 best = best := by
  rw [innerTail169_14, scan, if_neg (by decide), innerCandidates169_14_checked]
  change scan edge (search edge 3) 4 [1021] 1021 innerTail169_15
    (if 3 ≤ innerCandidates169_14.length then search edge 3 [1021, 6691] innerCandidates169_14 7712 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan169_15

def innerTail169_13 : List Nat := 6397 :: innerTail169_14
def innerCandidates169_13 : List Nat := [7213]
set_option maxRecDepth 4096 in
theorem innerCandidates169_13_checked : candidates edge 4 1021 6397 innerTail169_14 best.bound = innerCandidates169_13 := by
  unfold candidates
  have row : edge 6397 = SearchData.row6397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan169_13 : scan edge (search edge 3) 4 [1021] 1021 innerTail169_13 best = best := by
  rw [innerTail169_13, scan, if_neg (by decide), innerCandidates169_13_checked]
  change scan edge (search edge 3) 4 [1021] 1021 innerTail169_14
    (if 3 ≤ innerCandidates169_13.length then search edge 3 [1021, 6397] innerCandidates169_13 7418 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan169_14

def innerTail169_12 : List Nat := 6373 :: innerTail169_13
def innerCandidates169_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates169_12_checked : candidates edge 4 1021 6373 innerTail169_13 best.bound = innerCandidates169_12 := by
  unfold candidates
  have row : edge 6373 = SearchData.row6373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan169_12 : scan edge (search edge 3) 4 [1021] 1021 innerTail169_12 best = best := by
  rw [innerTail169_12, scan, if_neg (by decide), innerCandidates169_12_checked]
  change scan edge (search edge 3) 4 [1021] 1021 innerTail169_13
    (if 3 ≤ innerCandidates169_12.length then search edge 3 [1021, 6373] innerCandidates169_12 7394 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan169_13

def innerTail169_11 : List Nat := 6247 :: innerTail169_12
def innerCandidates169_11 : List Nat := [6691]
set_option maxRecDepth 4096 in
theorem innerCandidates169_11_checked : candidates edge 4 1021 6247 innerTail169_12 best.bound = innerCandidates169_11 := by
  unfold candidates
  have row : edge 6247 = SearchData.row6247.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan169_11 : scan edge (search edge 3) 4 [1021] 1021 innerTail169_11 best = best := by
  rw [innerTail169_11, scan, if_neg (by decide), innerCandidates169_11_checked]
  change scan edge (search edge 3) 4 [1021] 1021 innerTail169_12
    (if 3 ≤ innerCandidates169_11.length then search edge 3 [1021, 6247] innerCandidates169_11 7268 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan169_12

def innerTail169_10 : List Nat := 6079 :: innerTail169_11
def innerCandidates169_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates169_10_checked : candidates edge 4 1021 6079 innerTail169_11 best.bound = innerCandidates169_10 := by
  unfold candidates
  have row : edge 6079 = SearchData.row6079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan169_10 : scan edge (search edge 3) 4 [1021] 1021 innerTail169_10 best = best := by
  rw [innerTail169_10, scan, if_neg (by decide), innerCandidates169_10_checked]
  change scan edge (search edge 3) 4 [1021] 1021 innerTail169_11
    (if 3 ≤ innerCandidates169_10.length then search edge 3 [1021, 6079] innerCandidates169_10 7100 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan169_11

def innerTail169_9 : List Nat := 5701 :: innerTail169_10
def innerCandidates169_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates169_9_checked : candidates edge 4 1021 5701 innerTail169_10 best.bound = innerCandidates169_9 := by
  unfold candidates
  have row : edge 5701 = SearchData.row5701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan169_9 : scan edge (search edge 3) 4 [1021] 1021 innerTail169_9 best = best := by
  rw [innerTail169_9, scan, if_neg (by decide), innerCandidates169_9_checked]
  change scan edge (search edge 3) 4 [1021] 1021 innerTail169_10
    (if 3 ≤ innerCandidates169_9.length then search edge 3 [1021, 5701] innerCandidates169_9 6722 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan169_10

def innerTail169_8 : List Nat := 5281 :: innerTail169_9
def innerCandidates169_8 : List Nat := [7873]
set_option maxRecDepth 4096 in
theorem innerCandidates169_8_checked : candidates edge 4 1021 5281 innerTail169_9 best.bound = innerCandidates169_8 := by
  unfold candidates
  have row : edge 5281 = SearchData.row5281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan169_8 : scan edge (search edge 3) 4 [1021] 1021 innerTail169_8 best = best := by
  rw [innerTail169_8, scan, if_neg (by decide), innerCandidates169_8_checked]
  change scan edge (search edge 3) 4 [1021] 1021 innerTail169_9
    (if 3 ≤ innerCandidates169_8.length then search edge 3 [1021, 5281] innerCandidates169_8 6302 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan169_9

def innerTail169_7 : List Nat := 4597 :: innerTail169_8
def innerCandidates169_7 : List Nat := [6079, 7213]
set_option maxRecDepth 4096 in
theorem innerCandidates169_7_checked : candidates edge 4 1021 4597 innerTail169_8 best.bound = innerCandidates169_7 := by
  unfold candidates
  have row : edge 4597 = SearchData.row4597.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan169_7 : scan edge (search edge 3) 4 [1021] 1021 innerTail169_7 best = best := by
  rw [innerTail169_7, scan, if_neg (by decide), innerCandidates169_7_checked]
  change scan edge (search edge 3) 4 [1021] 1021 innerTail169_8
    (if 3 ≤ innerCandidates169_7.length then search edge 3 [1021, 4597] innerCandidates169_7 5618 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan169_8

def innerTail169_6 : List Nat := 3769 :: innerTail169_7
def innerCandidates169_6 : List Nat := [4597]
set_option maxRecDepth 4096 in
theorem innerCandidates169_6_checked : candidates edge 4 1021 3769 innerTail169_7 best.bound = innerCandidates169_6 := by
  unfold candidates
  have row : edge 3769 = SearchData.row3769.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan169_6 : scan edge (search edge 3) 4 [1021] 1021 innerTail169_6 best = best := by
  rw [innerTail169_6, scan, if_neg (by decide), innerCandidates169_6_checked]
  change scan edge (search edge 3) 4 [1021] 1021 innerTail169_7
    (if 3 ≤ innerCandidates169_6.length then search edge 3 [1021, 3769] innerCandidates169_6 4790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan169_7

def innerTail169_5 : List Nat := 3571 :: innerTail169_6
def innerCandidates169_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates169_5_checked : candidates edge 4 1021 3571 innerTail169_6 best.bound = innerCandidates169_5 := by
  unfold candidates
  have row : edge 3571 = SearchData.row3571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan169_5 : scan edge (search edge 3) 4 [1021] 1021 innerTail169_5 best = best := by
  rw [innerTail169_5, scan, if_neg (by decide), innerCandidates169_5_checked]
  change scan edge (search edge 3) 4 [1021] 1021 innerTail169_6
    (if 3 ≤ innerCandidates169_5.length then search edge 3 [1021, 3571] innerCandidates169_5 4592 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan169_6

def innerTail169_4 : List Nat := 3559 :: innerTail169_5
def innerCandidates169_4 : List Nat := [3769]
set_option maxRecDepth 4096 in
theorem innerCandidates169_4_checked : candidates edge 4 1021 3559 innerTail169_5 best.bound = innerCandidates169_4 := by
  unfold candidates
  have row : edge 3559 = SearchData.row3559.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan169_4 : scan edge (search edge 3) 4 [1021] 1021 innerTail169_4 best = best := by
  rw [innerTail169_4, scan, if_neg (by decide), innerCandidates169_4_checked]
  change scan edge (search edge 3) 4 [1021] 1021 innerTail169_5
    (if 3 ≤ innerCandidates169_4.length then search edge 3 [1021, 3559] innerCandidates169_4 4580 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan169_5

def innerTail169_3 : List Nat := 3529 :: innerTail169_4
def innerCandidates169_3 : List Nat := [6397]
set_option maxRecDepth 4096 in
theorem innerCandidates169_3_checked : candidates edge 4 1021 3529 innerTail169_4 best.bound = innerCandidates169_3 := by
  unfold candidates
  have row : edge 3529 = SearchData.row3529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan169_3 : scan edge (search edge 3) 4 [1021] 1021 innerTail169_3 best = best := by
  rw [innerTail169_3, scan, if_neg (by decide), innerCandidates169_3_checked]
  change scan edge (search edge 3) 4 [1021] 1021 innerTail169_4
    (if 3 ≤ innerCandidates169_3.length then search edge 3 [1021, 3529] innerCandidates169_3 4550 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan169_4

def innerTail169_2 : List Nat := 2749 :: innerTail169_3
def innerCandidates169_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates169_2_checked : candidates edge 4 1021 2749 innerTail169_3 best.bound = innerCandidates169_2 := by
  unfold candidates
  have row : edge 2749 = SearchData.row2749.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan169_2 : scan edge (search edge 3) 4 [1021] 1021 innerTail169_2 best = best := by
  rw [innerTail169_2, scan, if_neg (by decide), innerCandidates169_2_checked]
  change scan edge (search edge 3) 4 [1021] 1021 innerTail169_3
    (if 3 ≤ innerCandidates169_2.length then search edge 3 [1021, 2749] innerCandidates169_2 3770 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan169_3

def innerTail169_1 : List Nat := 2539 :: innerTail169_2
def innerCandidates169_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates169_1_checked : candidates edge 4 1021 2539 innerTail169_2 best.bound = innerCandidates169_1 := by
  unfold candidates
  have row : edge 2539 = PrimePairCache.smallRow2539.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan169_1 : scan edge (search edge 3) 4 [1021] 1021 innerTail169_1 best = best := by
  rw [innerTail169_1, scan, if_neg (by decide), innerCandidates169_1_checked]
  change scan edge (search edge 3) 4 [1021] 1021 innerTail169_2
    (if 3 ≤ innerCandidates169_1.length then search edge 3 [1021, 2539] innerCandidates169_1 3560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan169_2

def innerTail169_0 : List Nat := 1699 :: innerTail169_1
def innerCandidates169_0 : List Nat := [6691]
set_option maxRecDepth 4096 in
theorem innerCandidates169_0_checked : candidates edge 4 1021 1699 innerTail169_1 best.bound = innerCandidates169_0 := by
  unfold candidates
  have row : edge 1699 = PrimePairCache.smallRow1699.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan169_0 : scan edge (search edge 3) 4 [1021] 1021 innerTail169_0 best = best := by
  rw [innerTail169_0, scan, if_neg (by decide), innerCandidates169_0_checked]
  change scan edge (search edge 3) 4 [1021] 1021 innerTail169_1
    (if 3 ≤ innerCandidates169_0.length then search edge 3 [1021, 1699] innerCandidates169_0 2720 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan169_1

theorem rootBranch169 : search edge 4 [1021] rootCandidates169 1021 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1021] 1021 innerTail169_0 best = best
  exact innerScan169_0

def rootCandidates168 : List Nat := [1439, 2549, 2621, 2957, 2999, 3221, 3767, 3797, 3929, 4001, 4049, 4337, 4517, 4931, 5477, 5483, 5741, 6947, 7193, 7673, 7907, 7949]
set_option maxRecDepth 4096 in
theorem rootCandidates168_checked : candidates edge 5 0 1019 tail169 best.bound = rootCandidates168 := by
  unfold candidates
  have row : edge 1019 = PrimePairCache.smallRow1019.contains := by funext q; rfl
  rw [row]
  decide

def innerTail168_22 : List Nat := []
theorem innerScan168_22 : scan edge (search edge 3) 4 [1019] 1019 innerTail168_22 best = best := rfl

def innerTail168_21 : List Nat := 7949 :: innerTail168_22
def innerCandidates168_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates168_21_checked : candidates edge 4 1019 7949 innerTail168_22 best.bound = innerCandidates168_21 := by
  unfold candidates
  have row : edge 7949 = SearchData.row7949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan168_21 : scan edge (search edge 3) 4 [1019] 1019 innerTail168_21 best = best := by
  rw [innerTail168_21, scan, if_neg (by decide), innerCandidates168_21_checked]
  change scan edge (search edge 3) 4 [1019] 1019 innerTail168_22
    (if 3 ≤ innerCandidates168_21.length then search edge 3 [1019, 7949] innerCandidates168_21 8968 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan168_22

def innerTail168_20 : List Nat := 7907 :: innerTail168_21
def innerCandidates168_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates168_20_checked : candidates edge 4 1019 7907 innerTail168_21 best.bound = innerCandidates168_20 := by
  unfold candidates
  have row : edge 7907 = SearchData.row7907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan168_20 : scan edge (search edge 3) 4 [1019] 1019 innerTail168_20 best = best := by
  rw [innerTail168_20, scan, if_neg (by decide), innerCandidates168_20_checked]
  change scan edge (search edge 3) 4 [1019] 1019 innerTail168_21
    (if 3 ≤ innerCandidates168_20.length then search edge 3 [1019, 7907] innerCandidates168_20 8926 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan168_21

def innerTail168_19 : List Nat := 7673 :: innerTail168_20
def innerCandidates168_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates168_19_checked : candidates edge 4 1019 7673 innerTail168_20 best.bound = innerCandidates168_19 := by
  unfold candidates
  have row : edge 7673 = SearchData.row7673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan168_19 : scan edge (search edge 3) 4 [1019] 1019 innerTail168_19 best = best := by
  rw [innerTail168_19, scan, if_neg (by decide), innerCandidates168_19_checked]
  change scan edge (search edge 3) 4 [1019] 1019 innerTail168_20
    (if 3 ≤ innerCandidates168_19.length then search edge 3 [1019, 7673] innerCandidates168_19 8692 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan168_20

def innerTail168_18 : List Nat := 7193 :: innerTail168_19
def innerCandidates168_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates168_18_checked : candidates edge 4 1019 7193 innerTail168_19 best.bound = innerCandidates168_18 := by
  unfold candidates
  have row : edge 7193 = SearchData.row7193.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan168_18 : scan edge (search edge 3) 4 [1019] 1019 innerTail168_18 best = best := by
  rw [innerTail168_18, scan, if_neg (by decide), innerCandidates168_18_checked]
  change scan edge (search edge 3) 4 [1019] 1019 innerTail168_19
    (if 3 ≤ innerCandidates168_18.length then search edge 3 [1019, 7193] innerCandidates168_18 8212 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan168_19

def innerTail168_17 : List Nat := 6947 :: innerTail168_18
def innerCandidates168_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates168_17_checked : candidates edge 4 1019 6947 innerTail168_18 best.bound = innerCandidates168_17 := by
  unfold candidates
  have row : edge 6947 = SearchData.row6947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan168_17 : scan edge (search edge 3) 4 [1019] 1019 innerTail168_17 best = best := by
  rw [innerTail168_17, scan, if_neg (by decide), innerCandidates168_17_checked]
  change scan edge (search edge 3) 4 [1019] 1019 innerTail168_18
    (if 3 ≤ innerCandidates168_17.length then search edge 3 [1019, 6947] innerCandidates168_17 7966 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan168_18

def innerTail168_16 : List Nat := 5741 :: innerTail168_17
def innerCandidates168_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates168_16_checked : candidates edge 4 1019 5741 innerTail168_17 best.bound = innerCandidates168_16 := by
  unfold candidates
  have row : edge 5741 = SearchData.row5741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan168_16 : scan edge (search edge 3) 4 [1019] 1019 innerTail168_16 best = best := by
  rw [innerTail168_16, scan, if_neg (by decide), innerCandidates168_16_checked]
  change scan edge (search edge 3) 4 [1019] 1019 innerTail168_17
    (if 3 ≤ innerCandidates168_16.length then search edge 3 [1019, 5741] innerCandidates168_16 6760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan168_17

def innerTail168_15 : List Nat := 5483 :: innerTail168_16
def innerCandidates168_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates168_15_checked : candidates edge 4 1019 5483 innerTail168_16 best.bound = innerCandidates168_15 := by
  unfold candidates
  have row : edge 5483 = SearchData.row5483.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan168_15 : scan edge (search edge 3) 4 [1019] 1019 innerTail168_15 best = best := by
  rw [innerTail168_15, scan, if_neg (by decide), innerCandidates168_15_checked]
  change scan edge (search edge 3) 4 [1019] 1019 innerTail168_16
    (if 3 ≤ innerCandidates168_15.length then search edge 3 [1019, 5483] innerCandidates168_15 6502 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan168_16

def innerTail168_14 : List Nat := 5477 :: innerTail168_15
def innerCandidates168_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates168_14_checked : candidates edge 4 1019 5477 innerTail168_15 best.bound = innerCandidates168_14 := by
  unfold candidates
  have row : edge 5477 = SearchData.row5477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan168_14 : scan edge (search edge 3) 4 [1019] 1019 innerTail168_14 best = best := by
  rw [innerTail168_14, scan, if_neg (by decide), innerCandidates168_14_checked]
  change scan edge (search edge 3) 4 [1019] 1019 innerTail168_15
    (if 3 ≤ innerCandidates168_14.length then search edge 3 [1019, 5477] innerCandidates168_14 6496 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan168_15

def innerTail168_13 : List Nat := 4931 :: innerTail168_14
def innerCandidates168_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates168_13_checked : candidates edge 4 1019 4931 innerTail168_14 best.bound = innerCandidates168_13 := by
  unfold candidates
  have row : edge 4931 = SearchData.row4931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan168_13 : scan edge (search edge 3) 4 [1019] 1019 innerTail168_13 best = best := by
  rw [innerTail168_13, scan, if_neg (by decide), innerCandidates168_13_checked]
  change scan edge (search edge 3) 4 [1019] 1019 innerTail168_14
    (if 3 ≤ innerCandidates168_13.length then search edge 3 [1019, 4931] innerCandidates168_13 5950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan168_14

def innerTail168_12 : List Nat := 4517 :: innerTail168_13
def innerCandidates168_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates168_12_checked : candidates edge 4 1019 4517 innerTail168_13 best.bound = innerCandidates168_12 := by
  unfold candidates
  have row : edge 4517 = SearchData.row4517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan168_12 : scan edge (search edge 3) 4 [1019] 1019 innerTail168_12 best = best := by
  rw [innerTail168_12, scan, if_neg (by decide), innerCandidates168_12_checked]
  change scan edge (search edge 3) 4 [1019] 1019 innerTail168_13
    (if 3 ≤ innerCandidates168_12.length then search edge 3 [1019, 4517] innerCandidates168_12 5536 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan168_13

def innerTail168_11 : List Nat := 4337 :: innerTail168_12
def innerCandidates168_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates168_11_checked : candidates edge 4 1019 4337 innerTail168_12 best.bound = innerCandidates168_11 := by
  unfold candidates
  have row : edge 4337 = SearchData.row4337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan168_11 : scan edge (search edge 3) 4 [1019] 1019 innerTail168_11 best = best := by
  rw [innerTail168_11, scan, if_neg (by decide), innerCandidates168_11_checked]
  change scan edge (search edge 3) 4 [1019] 1019 innerTail168_12
    (if 3 ≤ innerCandidates168_11.length then search edge 3 [1019, 4337] innerCandidates168_11 5356 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan168_12

def innerTail168_10 : List Nat := 4049 :: innerTail168_11
def innerCandidates168_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates168_10_checked : candidates edge 4 1019 4049 innerTail168_11 best.bound = innerCandidates168_10 := by
  unfold candidates
  have row : edge 4049 = SearchData.row4049.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan168_10 : scan edge (search edge 3) 4 [1019] 1019 innerTail168_10 best = best := by
  rw [innerTail168_10, scan, if_neg (by decide), innerCandidates168_10_checked]
  change scan edge (search edge 3) 4 [1019] 1019 innerTail168_11
    (if 3 ≤ innerCandidates168_10.length then search edge 3 [1019, 4049] innerCandidates168_10 5068 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan168_11

def innerTail168_9 : List Nat := 4001 :: innerTail168_10
def innerCandidates168_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates168_9_checked : candidates edge 4 1019 4001 innerTail168_10 best.bound = innerCandidates168_9 := by
  unfold candidates
  have row : edge 4001 = SearchData.row4001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan168_9 : scan edge (search edge 3) 4 [1019] 1019 innerTail168_9 best = best := by
  rw [innerTail168_9, scan, if_neg (by decide), innerCandidates168_9_checked]
  change scan edge (search edge 3) 4 [1019] 1019 innerTail168_10
    (if 3 ≤ innerCandidates168_9.length then search edge 3 [1019, 4001] innerCandidates168_9 5020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan168_10

def innerTail168_8 : List Nat := 3929 :: innerTail168_9
def innerCandidates168_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates168_8_checked : candidates edge 4 1019 3929 innerTail168_9 best.bound = innerCandidates168_8 := by
  unfold candidates
  have row : edge 3929 = SearchData.row3929.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan168_8 : scan edge (search edge 3) 4 [1019] 1019 innerTail168_8 best = best := by
  rw [innerTail168_8, scan, if_neg (by decide), innerCandidates168_8_checked]
  change scan edge (search edge 3) 4 [1019] 1019 innerTail168_9
    (if 3 ≤ innerCandidates168_8.length then search edge 3 [1019, 3929] innerCandidates168_8 4948 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan168_9

def innerTail168_7 : List Nat := 3797 :: innerTail168_8
def innerCandidates168_7 : List Nat := [4049]
set_option maxRecDepth 4096 in
theorem innerCandidates168_7_checked : candidates edge 4 1019 3797 innerTail168_8 best.bound = innerCandidates168_7 := by
  unfold candidates
  have row : edge 3797 = SearchData.row3797.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan168_7 : scan edge (search edge 3) 4 [1019] 1019 innerTail168_7 best = best := by
  rw [innerTail168_7, scan, if_neg (by decide), innerCandidates168_7_checked]
  change scan edge (search edge 3) 4 [1019] 1019 innerTail168_8
    (if 3 ≤ innerCandidates168_7.length then search edge 3 [1019, 3797] innerCandidates168_7 4816 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan168_8

def innerTail168_6 : List Nat := 3767 :: innerTail168_7
def innerCandidates168_6 : List Nat := [7193]
set_option maxRecDepth 4096 in
theorem innerCandidates168_6_checked : candidates edge 4 1019 3767 innerTail168_7 best.bound = innerCandidates168_6 := by
  unfold candidates
  have row : edge 3767 = SearchData.row3767.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan168_6 : scan edge (search edge 3) 4 [1019] 1019 innerTail168_6 best = best := by
  rw [innerTail168_6, scan, if_neg (by decide), innerCandidates168_6_checked]
  change scan edge (search edge 3) 4 [1019] 1019 innerTail168_7
    (if 3 ≤ innerCandidates168_6.length then search edge 3 [1019, 3767] innerCandidates168_6 4786 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan168_7

def innerTail168_5 : List Nat := 3221 :: innerTail168_6
def innerCandidates168_5 : List Nat := [3767]
set_option maxRecDepth 4096 in
theorem innerCandidates168_5_checked : candidates edge 4 1019 3221 innerTail168_6 best.bound = innerCandidates168_5 := by
  unfold candidates
  have row : edge 3221 = SearchData.row3221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan168_5 : scan edge (search edge 3) 4 [1019] 1019 innerTail168_5 best = best := by
  rw [innerTail168_5, scan, if_neg (by decide), innerCandidates168_5_checked]
  change scan edge (search edge 3) 4 [1019] 1019 innerTail168_6
    (if 3 ≤ innerCandidates168_5.length then search edge 3 [1019, 3221] innerCandidates168_5 4240 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan168_6

def innerTail168_4 : List Nat := 2999 :: innerTail168_5
def innerCandidates168_4 : List Nat := [7907]
set_option maxRecDepth 4096 in
theorem innerCandidates168_4_checked : candidates edge 4 1019 2999 innerTail168_5 best.bound = innerCandidates168_4 := by
  unfold candidates
  have row : edge 2999 = SearchData.row2999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan168_4 : scan edge (search edge 3) 4 [1019] 1019 innerTail168_4 best = best := by
  rw [innerTail168_4, scan, if_neg (by decide), innerCandidates168_4_checked]
  change scan edge (search edge 3) 4 [1019] 1019 innerTail168_5
    (if 3 ≤ innerCandidates168_4.length then search edge 3 [1019, 2999] innerCandidates168_4 4018 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan168_5

def innerTail168_3 : List Nat := 2957 :: innerTail168_4
def innerCandidates168_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates168_3_checked : candidates edge 4 1019 2957 innerTail168_4 best.bound = innerCandidates168_3 := by
  unfold candidates
  have row : edge 2957 = SearchData.row2957.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan168_3 : scan edge (search edge 3) 4 [1019] 1019 innerTail168_3 best = best := by
  rw [innerTail168_3, scan, if_neg (by decide), innerCandidates168_3_checked]
  change scan edge (search edge 3) 4 [1019] 1019 innerTail168_4
    (if 3 ≤ innerCandidates168_3.length then search edge 3 [1019, 2957] innerCandidates168_3 3976 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan168_4

def innerTail168_2 : List Nat := 2621 :: innerTail168_3
def innerCandidates168_2 : List Nat := [5477]
set_option maxRecDepth 4096 in
theorem innerCandidates168_2_checked : candidates edge 4 1019 2621 innerTail168_3 best.bound = innerCandidates168_2 := by
  unfold candidates
  have row : edge 2621 = PrimePairCache.smallRow2621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan168_2 : scan edge (search edge 3) 4 [1019] 1019 innerTail168_2 best = best := by
  rw [innerTail168_2, scan, if_neg (by decide), innerCandidates168_2_checked]
  change scan edge (search edge 3) 4 [1019] 1019 innerTail168_3
    (if 3 ≤ innerCandidates168_2.length then search edge 3 [1019, 2621] innerCandidates168_2 3640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan168_3

def innerTail168_1 : List Nat := 2549 :: innerTail168_2
def innerCandidates168_1 : List Nat := [7673, 7949]
set_option maxRecDepth 4096 in
theorem innerCandidates168_1_checked : candidates edge 4 1019 2549 innerTail168_2 best.bound = innerCandidates168_1 := by
  unfold candidates
  have row : edge 2549 = PrimePairCache.smallRow2549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan168_1 : scan edge (search edge 3) 4 [1019] 1019 innerTail168_1 best = best := by
  rw [innerTail168_1, scan, if_neg (by decide), innerCandidates168_1_checked]
  change scan edge (search edge 3) 4 [1019] 1019 innerTail168_2
    (if 3 ≤ innerCandidates168_1.length then search edge 3 [1019, 2549] innerCandidates168_1 3568 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan168_2

def innerTail168_0 : List Nat := 1439 :: innerTail168_1
def innerCandidates168_0 : List Nat := [4337, 4931]
set_option maxRecDepth 4096 in
theorem innerCandidates168_0_checked : candidates edge 4 1019 1439 innerTail168_1 best.bound = innerCandidates168_0 := by
  unfold candidates
  have row : edge 1439 = PrimePairCache.smallRow1439.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan168_0 : scan edge (search edge 3) 4 [1019] 1019 innerTail168_0 best = best := by
  rw [innerTail168_0, scan, if_neg (by decide), innerCandidates168_0_checked]
  change scan edge (search edge 3) 4 [1019] 1019 innerTail168_1
    (if 3 ≤ innerCandidates168_0.length then search edge 3 [1019, 1439] innerCandidates168_0 2458 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan168_1

theorem rootBranch168 : search edge 4 [1019] rootCandidates168 1019 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1019] 1019 innerTail168_0 best = best
  exact innerScan168_0

end ProjectEulerSolutions.PrimePairUpper.Search
