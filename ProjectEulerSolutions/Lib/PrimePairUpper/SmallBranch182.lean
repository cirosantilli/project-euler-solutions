import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates143 : List Nat := [887, 911, 1109, 1283, 1289, 1949, 2063, 2333, 3251, 3533, 3923, 4679, 4721, 5237, 5273, 5351, 5741, 6263, 6599, 7121, 7283, 7331, 7457, 7841, 7877, 8387]
set_option maxRecDepth 4096 in
theorem rootCandidates143_checked : candidates edge 5 0 839 tail144 best.bound = rootCandidates143 := by
  unfold candidates
  have row : edge 839 = PrimePairCache.smallRow839.contains := by funext q; rfl
  rw [row]
  decide

def innerTail143_26 : List Nat := []
theorem innerScan143_26 : scan edge (search edge 3) 4 [839] 839 innerTail143_26 best = best := rfl

def innerTail143_25 : List Nat := 8387 :: innerTail143_26
def innerCandidates143_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates143_25_checked : candidates edge 4 839 8387 innerTail143_26 best.bound = innerCandidates143_25 := by
  unfold candidates
  have row : edge 8387 = SearchData.row8387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan143_25 : scan edge (search edge 3) 4 [839] 839 innerTail143_25 best = best := by
  rw [innerTail143_25, scan, if_neg (by decide), innerCandidates143_25_checked]
  change scan edge (search edge 3) 4 [839] 839 innerTail143_26
    (if 3 ≤ innerCandidates143_25.length then search edge 3 [839, 8387] innerCandidates143_25 9226 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan143_26

def innerTail143_24 : List Nat := 7877 :: innerTail143_25
def innerCandidates143_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates143_24_checked : candidates edge 4 839 7877 innerTail143_25 best.bound = innerCandidates143_24 := by
  unfold candidates
  have row : edge 7877 = SearchData.row7877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan143_24 : scan edge (search edge 3) 4 [839] 839 innerTail143_24 best = best := by
  rw [innerTail143_24, scan, if_neg (by decide), innerCandidates143_24_checked]
  change scan edge (search edge 3) 4 [839] 839 innerTail143_25
    (if 3 ≤ innerCandidates143_24.length then search edge 3 [839, 7877] innerCandidates143_24 8716 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan143_25

def innerTail143_23 : List Nat := 7841 :: innerTail143_24
def innerCandidates143_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates143_23_checked : candidates edge 4 839 7841 innerTail143_24 best.bound = innerCandidates143_23 := by
  unfold candidates
  have row : edge 7841 = SearchData.row7841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan143_23 : scan edge (search edge 3) 4 [839] 839 innerTail143_23 best = best := by
  rw [innerTail143_23, scan, if_neg (by decide), innerCandidates143_23_checked]
  change scan edge (search edge 3) 4 [839] 839 innerTail143_24
    (if 3 ≤ innerCandidates143_23.length then search edge 3 [839, 7841] innerCandidates143_23 8680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan143_24

def innerTail143_22 : List Nat := 7457 :: innerTail143_23
def innerCandidates143_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates143_22_checked : candidates edge 4 839 7457 innerTail143_23 best.bound = innerCandidates143_22 := by
  unfold candidates
  have row : edge 7457 = SearchData.row7457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan143_22 : scan edge (search edge 3) 4 [839] 839 innerTail143_22 best = best := by
  rw [innerTail143_22, scan, if_neg (by decide), innerCandidates143_22_checked]
  change scan edge (search edge 3) 4 [839] 839 innerTail143_23
    (if 3 ≤ innerCandidates143_22.length then search edge 3 [839, 7457] innerCandidates143_22 8296 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan143_23

def innerTail143_21 : List Nat := 7331 :: innerTail143_22
def innerCandidates143_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates143_21_checked : candidates edge 4 839 7331 innerTail143_22 best.bound = innerCandidates143_21 := by
  unfold candidates
  have row : edge 7331 = SearchData.row7331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan143_21 : scan edge (search edge 3) 4 [839] 839 innerTail143_21 best = best := by
  rw [innerTail143_21, scan, if_neg (by decide), innerCandidates143_21_checked]
  change scan edge (search edge 3) 4 [839] 839 innerTail143_22
    (if 3 ≤ innerCandidates143_21.length then search edge 3 [839, 7331] innerCandidates143_21 8170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan143_22

def innerTail143_20 : List Nat := 7283 :: innerTail143_21
def innerCandidates143_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates143_20_checked : candidates edge 4 839 7283 innerTail143_21 best.bound = innerCandidates143_20 := by
  unfold candidates
  have row : edge 7283 = SearchData.row7283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan143_20 : scan edge (search edge 3) 4 [839] 839 innerTail143_20 best = best := by
  rw [innerTail143_20, scan, if_neg (by decide), innerCandidates143_20_checked]
  change scan edge (search edge 3) 4 [839] 839 innerTail143_21
    (if 3 ≤ innerCandidates143_20.length then search edge 3 [839, 7283] innerCandidates143_20 8122 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan143_21

def innerTail143_19 : List Nat := 7121 :: innerTail143_20
def innerCandidates143_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates143_19_checked : candidates edge 4 839 7121 innerTail143_20 best.bound = innerCandidates143_19 := by
  unfold candidates
  have row : edge 7121 = SearchData.row7121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan143_19 : scan edge (search edge 3) 4 [839] 839 innerTail143_19 best = best := by
  rw [innerTail143_19, scan, if_neg (by decide), innerCandidates143_19_checked]
  change scan edge (search edge 3) 4 [839] 839 innerTail143_20
    (if 3 ≤ innerCandidates143_19.length then search edge 3 [839, 7121] innerCandidates143_19 7960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan143_20

def innerTail143_18 : List Nat := 6599 :: innerTail143_19
def innerCandidates143_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates143_18_checked : candidates edge 4 839 6599 innerTail143_19 best.bound = innerCandidates143_18 := by
  unfold candidates
  have row : edge 6599 = SearchData.row6599.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan143_18 : scan edge (search edge 3) 4 [839] 839 innerTail143_18 best = best := by
  rw [innerTail143_18, scan, if_neg (by decide), innerCandidates143_18_checked]
  change scan edge (search edge 3) 4 [839] 839 innerTail143_19
    (if 3 ≤ innerCandidates143_18.length then search edge 3 [839, 6599] innerCandidates143_18 7438 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan143_19

def innerTail143_17 : List Nat := 6263 :: innerTail143_18
def innerCandidates143_17 : List Nat := [7121]
set_option maxRecDepth 4096 in
theorem innerCandidates143_17_checked : candidates edge 4 839 6263 innerTail143_18 best.bound = innerCandidates143_17 := by
  unfold candidates
  have row : edge 6263 = SearchData.row6263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan143_17 : scan edge (search edge 3) 4 [839] 839 innerTail143_17 best = best := by
  rw [innerTail143_17, scan, if_neg (by decide), innerCandidates143_17_checked]
  change scan edge (search edge 3) 4 [839] 839 innerTail143_18
    (if 3 ≤ innerCandidates143_17.length then search edge 3 [839, 6263] innerCandidates143_17 7102 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan143_18

def innerTail143_16 : List Nat := 5741 :: innerTail143_17
def innerCandidates143_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates143_16_checked : candidates edge 4 839 5741 innerTail143_17 best.bound = innerCandidates143_16 := by
  unfold candidates
  have row : edge 5741 = SearchData.row5741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan143_16 : scan edge (search edge 3) 4 [839] 839 innerTail143_16 best = best := by
  rw [innerTail143_16, scan, if_neg (by decide), innerCandidates143_16_checked]
  change scan edge (search edge 3) 4 [839] 839 innerTail143_17
    (if 3 ≤ innerCandidates143_16.length then search edge 3 [839, 5741] innerCandidates143_16 6580 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan143_17

def innerTail143_15 : List Nat := 5351 :: innerTail143_16
def innerCandidates143_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates143_15_checked : candidates edge 4 839 5351 innerTail143_16 best.bound = innerCandidates143_15 := by
  unfold candidates
  have row : edge 5351 = SearchData.row5351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan143_15 : scan edge (search edge 3) 4 [839] 839 innerTail143_15 best = best := by
  rw [innerTail143_15, scan, if_neg (by decide), innerCandidates143_15_checked]
  change scan edge (search edge 3) 4 [839] 839 innerTail143_16
    (if 3 ≤ innerCandidates143_15.length then search edge 3 [839, 5351] innerCandidates143_15 6190 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan143_16

def innerTail143_14 : List Nat := 5273 :: innerTail143_15
def innerCandidates143_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates143_14_checked : candidates edge 4 839 5273 innerTail143_15 best.bound = innerCandidates143_14 := by
  unfold candidates
  have row : edge 5273 = SearchData.row5273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan143_14 : scan edge (search edge 3) 4 [839] 839 innerTail143_14 best = best := by
  rw [innerTail143_14, scan, if_neg (by decide), innerCandidates143_14_checked]
  change scan edge (search edge 3) 4 [839] 839 innerTail143_15
    (if 3 ≤ innerCandidates143_14.length then search edge 3 [839, 5273] innerCandidates143_14 6112 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan143_15

def innerTail143_13 : List Nat := 5237 :: innerTail143_14
def innerCandidates143_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates143_13_checked : candidates edge 4 839 5237 innerTail143_14 best.bound = innerCandidates143_13 := by
  unfold candidates
  have row : edge 5237 = SearchData.row5237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan143_13 : scan edge (search edge 3) 4 [839] 839 innerTail143_13 best = best := by
  rw [innerTail143_13, scan, if_neg (by decide), innerCandidates143_13_checked]
  change scan edge (search edge 3) 4 [839] 839 innerTail143_14
    (if 3 ≤ innerCandidates143_13.length then search edge 3 [839, 5237] innerCandidates143_13 6076 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan143_14

def innerTail143_12 : List Nat := 4721 :: innerTail143_13
def innerCandidates143_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates143_12_checked : candidates edge 4 839 4721 innerTail143_13 best.bound = innerCandidates143_12 := by
  unfold candidates
  have row : edge 4721 = SearchData.row4721.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan143_12 : scan edge (search edge 3) 4 [839] 839 innerTail143_12 best = best := by
  rw [innerTail143_12, scan, if_neg (by decide), innerCandidates143_12_checked]
  change scan edge (search edge 3) 4 [839] 839 innerTail143_13
    (if 3 ≤ innerCandidates143_12.length then search edge 3 [839, 4721] innerCandidates143_12 5560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan143_13

def innerTail143_11 : List Nat := 4679 :: innerTail143_12
def innerCandidates143_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates143_11_checked : candidates edge 4 839 4679 innerTail143_12 best.bound = innerCandidates143_11 := by
  unfold candidates
  have row : edge 4679 = SearchData.row4679.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan143_11 : scan edge (search edge 3) 4 [839] 839 innerTail143_11 best = best := by
  rw [innerTail143_11, scan, if_neg (by decide), innerCandidates143_11_checked]
  change scan edge (search edge 3) 4 [839] 839 innerTail143_12
    (if 3 ≤ innerCandidates143_11.length then search edge 3 [839, 4679] innerCandidates143_11 5518 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan143_12

def innerTail143_10 : List Nat := 3923 :: innerTail143_11
def innerCandidates143_10 : List Nat := [4721]
set_option maxRecDepth 4096 in
theorem innerCandidates143_10_checked : candidates edge 4 839 3923 innerTail143_11 best.bound = innerCandidates143_10 := by
  unfold candidates
  have row : edge 3923 = SearchData.row3923.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan143_10 : scan edge (search edge 3) 4 [839] 839 innerTail143_10 best = best := by
  rw [innerTail143_10, scan, if_neg (by decide), innerCandidates143_10_checked]
  change scan edge (search edge 3) 4 [839] 839 innerTail143_11
    (if 3 ≤ innerCandidates143_10.length then search edge 3 [839, 3923] innerCandidates143_10 4762 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan143_11

def innerTail143_9 : List Nat := 3533 :: innerTail143_10
def innerCandidates143_9 : List Nat := [4679]
set_option maxRecDepth 4096 in
theorem innerCandidates143_9_checked : candidates edge 4 839 3533 innerTail143_10 best.bound = innerCandidates143_9 := by
  unfold candidates
  have row : edge 3533 = SearchData.row3533.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan143_9 : scan edge (search edge 3) 4 [839] 839 innerTail143_9 best = best := by
  rw [innerTail143_9, scan, if_neg (by decide), innerCandidates143_9_checked]
  change scan edge (search edge 3) 4 [839] 839 innerTail143_10
    (if 3 ≤ innerCandidates143_9.length then search edge 3 [839, 3533] innerCandidates143_9 4372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan143_10

def innerTail143_8 : List Nat := 3251 :: innerTail143_9
def innerCandidates143_8 : List Nat := [4721]
set_option maxRecDepth 4096 in
theorem innerCandidates143_8_checked : candidates edge 4 839 3251 innerTail143_9 best.bound = innerCandidates143_8 := by
  unfold candidates
  have row : edge 3251 = SearchData.row3251.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan143_8 : scan edge (search edge 3) 4 [839] 839 innerTail143_8 best = best := by
  rw [innerTail143_8, scan, if_neg (by decide), innerCandidates143_8_checked]
  change scan edge (search edge 3) 4 [839] 839 innerTail143_9
    (if 3 ≤ innerCandidates143_8.length then search edge 3 [839, 3251] innerCandidates143_8 4090 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan143_9

def innerTail143_7 : List Nat := 2333 :: innerTail143_8
def innerCandidates143_7 : List Nat := [3533, 8387]
set_option maxRecDepth 4096 in
theorem innerCandidates143_7_checked : candidates edge 4 839 2333 innerTail143_8 best.bound = innerCandidates143_7 := by
  unfold candidates
  have row : edge 2333 = PrimePairCache.smallRow2333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan143_7 : scan edge (search edge 3) 4 [839] 839 innerTail143_7 best = best := by
  rw [innerTail143_7, scan, if_neg (by decide), innerCandidates143_7_checked]
  change scan edge (search edge 3) 4 [839] 839 innerTail143_8
    (if 3 ≤ innerCandidates143_7.length then search edge 3 [839, 2333] innerCandidates143_7 3172 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan143_8

def innerTail143_6 : List Nat := 2063 :: innerTail143_7
def innerCandidates143_6 : List Nat := [2333, 6599]
set_option maxRecDepth 4096 in
theorem innerCandidates143_6_checked : candidates edge 4 839 2063 innerTail143_7 best.bound = innerCandidates143_6 := by
  unfold candidates
  have row : edge 2063 = PrimePairCache.smallRow2063.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan143_6 : scan edge (search edge 3) 4 [839] 839 innerTail143_6 best = best := by
  rw [innerTail143_6, scan, if_neg (by decide), innerCandidates143_6_checked]
  change scan edge (search edge 3) 4 [839] 839 innerTail143_7
    (if 3 ≤ innerCandidates143_6.length then search edge 3 [839, 2063] innerCandidates143_6 2902 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan143_7

def innerTail143_5 : List Nat := 1949 :: innerTail143_6
def innerCandidates143_5 : List Nat := [3923]
set_option maxRecDepth 4096 in
theorem innerCandidates143_5_checked : candidates edge 4 839 1949 innerTail143_6 best.bound = innerCandidates143_5 := by
  unfold candidates
  have row : edge 1949 = PrimePairCache.smallRow1949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan143_5 : scan edge (search edge 3) 4 [839] 839 innerTail143_5 best = best := by
  rw [innerTail143_5, scan, if_neg (by decide), innerCandidates143_5_checked]
  change scan edge (search edge 3) 4 [839] 839 innerTail143_6
    (if 3 ≤ innerCandidates143_5.length then search edge 3 [839, 1949] innerCandidates143_5 2788 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan143_6

def innerTail143_4 : List Nat := 1289 :: innerTail143_5
def innerCandidates143_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates143_4_checked : candidates edge 4 839 1289 innerTail143_5 best.bound = innerCandidates143_4 := by
  unfold candidates
  have row : edge 1289 = PrimePairCache.smallRow1289.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan143_4 : scan edge (search edge 3) 4 [839] 839 innerTail143_4 best = best := by
  rw [innerTail143_4, scan, if_neg (by decide), innerCandidates143_4_checked]
  change scan edge (search edge 3) 4 [839] 839 innerTail143_5
    (if 3 ≤ innerCandidates143_4.length then search edge 3 [839, 1289] innerCandidates143_4 2128 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan143_5

def innerTail143_3 : List Nat := 1283 :: innerTail143_4
def innerCandidates143_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates143_3_checked : candidates edge 4 839 1283 innerTail143_4 best.bound = innerCandidates143_3 := by
  unfold candidates
  have row : edge 1283 = PrimePairCache.smallRow1283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan143_3 : scan edge (search edge 3) 4 [839] 839 innerTail143_3 best = best := by
  rw [innerTail143_3, scan, if_neg (by decide), innerCandidates143_3_checked]
  change scan edge (search edge 3) 4 [839] 839 innerTail143_4
    (if 3 ≤ innerCandidates143_3.length then search edge 3 [839, 1283] innerCandidates143_3 2122 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan143_4

def innerTail143_2 : List Nat := 1109 :: innerTail143_3
def innerCandidates143_2 : List Nat := [3923]
set_option maxRecDepth 4096 in
theorem innerCandidates143_2_checked : candidates edge 4 839 1109 innerTail143_3 best.bound = innerCandidates143_2 := by
  unfold candidates
  have row : edge 1109 = PrimePairCache.smallRow1109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan143_2 : scan edge (search edge 3) 4 [839] 839 innerTail143_2 best = best := by
  rw [innerTail143_2, scan, if_neg (by decide), innerCandidates143_2_checked]
  change scan edge (search edge 3) 4 [839] 839 innerTail143_3
    (if 3 ≤ innerCandidates143_2.length then search edge 3 [839, 1109] innerCandidates143_2 1948 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan143_3

def innerTail143_1 : List Nat := 911 :: innerTail143_2
def innerCandidates143_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates143_1_checked : candidates edge 4 839 911 innerTail143_2 best.bound = innerCandidates143_1 := by
  unfold candidates
  have row : edge 911 = PrimePairCache.smallRow911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan143_1 : scan edge (search edge 3) 4 [839] 839 innerTail143_1 best = best := by
  rw [innerTail143_1, scan, if_neg (by decide), innerCandidates143_1_checked]
  change scan edge (search edge 3) 4 [839] 839 innerTail143_2
    (if 3 ≤ innerCandidates143_1.length then search edge 3 [839, 911] innerCandidates143_1 1750 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan143_2

def innerTail143_0 : List Nat := 887 :: innerTail143_1
def innerCandidates143_0 : List Nat := [3251]
set_option maxRecDepth 4096 in
theorem innerCandidates143_0_checked : candidates edge 4 839 887 innerTail143_1 best.bound = innerCandidates143_0 := by
  unfold candidates
  have row : edge 887 = PrimePairCache.smallRow887.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan143_0 : scan edge (search edge 3) 4 [839] 839 innerTail143_0 best = best := by
  rw [innerTail143_0, scan, if_neg (by decide), innerCandidates143_0_checked]
  change scan edge (search edge 3) 4 [839] 839 innerTail143_1
    (if 3 ≤ innerCandidates143_0.length then search edge 3 [839, 887] innerCandidates143_0 1726 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan143_1

theorem rootBranch143 : search edge 4 [839] rootCandidates143 839 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [839] 839 innerTail143_0 best = best
  exact innerScan143_0

def rootCandidates142 : List Nat := [1033, 1237, 2293, 2671, 3319, 3361, 3847, 5107, 5527, 5563, 5839, 5857, 6823, 7321, 7351, 7573, 8209]
set_option maxRecDepth 4096 in
theorem rootCandidates142_checked : candidates edge 5 0 829 tail143 best.bound = rootCandidates142 := by
  unfold candidates
  have row : edge 829 = PrimePairCache.smallRow829.contains := by funext q; rfl
  rw [row]
  decide

def innerTail142_17 : List Nat := []
theorem innerScan142_17 : scan edge (search edge 3) 4 [829] 829 innerTail142_17 best = best := rfl

def innerTail142_16 : List Nat := 8209 :: innerTail142_17
def innerCandidates142_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates142_16_checked : candidates edge 4 829 8209 innerTail142_17 best.bound = innerCandidates142_16 := by
  unfold candidates
  have row : edge 8209 = SearchData.row8209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan142_16 : scan edge (search edge 3) 4 [829] 829 innerTail142_16 best = best := by
  rw [innerTail142_16, scan, if_neg (by decide), innerCandidates142_16_checked]
  change scan edge (search edge 3) 4 [829] 829 innerTail142_17
    (if 3 ≤ innerCandidates142_16.length then search edge 3 [829, 8209] innerCandidates142_16 9038 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan142_17

def innerTail142_15 : List Nat := 7573 :: innerTail142_16
def innerCandidates142_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates142_15_checked : candidates edge 4 829 7573 innerTail142_16 best.bound = innerCandidates142_15 := by
  unfold candidates
  have row : edge 7573 = SearchData.row7573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan142_15 : scan edge (search edge 3) 4 [829] 829 innerTail142_15 best = best := by
  rw [innerTail142_15, scan, if_neg (by decide), innerCandidates142_15_checked]
  change scan edge (search edge 3) 4 [829] 829 innerTail142_16
    (if 3 ≤ innerCandidates142_15.length then search edge 3 [829, 7573] innerCandidates142_15 8402 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan142_16

def innerTail142_14 : List Nat := 7351 :: innerTail142_15
def innerCandidates142_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates142_14_checked : candidates edge 4 829 7351 innerTail142_15 best.bound = innerCandidates142_14 := by
  unfold candidates
  have row : edge 7351 = SearchData.row7351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan142_14 : scan edge (search edge 3) 4 [829] 829 innerTail142_14 best = best := by
  rw [innerTail142_14, scan, if_neg (by decide), innerCandidates142_14_checked]
  change scan edge (search edge 3) 4 [829] 829 innerTail142_15
    (if 3 ≤ innerCandidates142_14.length then search edge 3 [829, 7351] innerCandidates142_14 8180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan142_15

def innerTail142_13 : List Nat := 7321 :: innerTail142_14
def innerCandidates142_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates142_13_checked : candidates edge 4 829 7321 innerTail142_14 best.bound = innerCandidates142_13 := by
  unfold candidates
  have row : edge 7321 = SearchData.row7321.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan142_13 : scan edge (search edge 3) 4 [829] 829 innerTail142_13 best = best := by
  rw [innerTail142_13, scan, if_neg (by decide), innerCandidates142_13_checked]
  change scan edge (search edge 3) 4 [829] 829 innerTail142_14
    (if 3 ≤ innerCandidates142_13.length then search edge 3 [829, 7321] innerCandidates142_13 8150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan142_14

def innerTail142_12 : List Nat := 6823 :: innerTail142_13
def innerCandidates142_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates142_12_checked : candidates edge 4 829 6823 innerTail142_13 best.bound = innerCandidates142_12 := by
  unfold candidates
  have row : edge 6823 = SearchData.row6823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan142_12 : scan edge (search edge 3) 4 [829] 829 innerTail142_12 best = best := by
  rw [innerTail142_12, scan, if_neg (by decide), innerCandidates142_12_checked]
  change scan edge (search edge 3) 4 [829] 829 innerTail142_13
    (if 3 ≤ innerCandidates142_12.length then search edge 3 [829, 6823] innerCandidates142_12 7652 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan142_13

def innerTail142_11 : List Nat := 5857 :: innerTail142_12
def innerCandidates142_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates142_11_checked : candidates edge 4 829 5857 innerTail142_12 best.bound = innerCandidates142_11 := by
  unfold candidates
  have row : edge 5857 = SearchData.row5857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan142_11 : scan edge (search edge 3) 4 [829] 829 innerTail142_11 best = best := by
  rw [innerTail142_11, scan, if_neg (by decide), innerCandidates142_11_checked]
  change scan edge (search edge 3) 4 [829] 829 innerTail142_12
    (if 3 ≤ innerCandidates142_11.length then search edge 3 [829, 5857] innerCandidates142_11 6686 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan142_12

def innerTail142_10 : List Nat := 5839 :: innerTail142_11
def innerCandidates142_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates142_10_checked : candidates edge 4 829 5839 innerTail142_11 best.bound = innerCandidates142_10 := by
  unfold candidates
  have row : edge 5839 = SearchData.row5839.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan142_10 : scan edge (search edge 3) 4 [829] 829 innerTail142_10 best = best := by
  rw [innerTail142_10, scan, if_neg (by decide), innerCandidates142_10_checked]
  change scan edge (search edge 3) 4 [829] 829 innerTail142_11
    (if 3 ≤ innerCandidates142_10.length then search edge 3 [829, 5839] innerCandidates142_10 6668 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan142_11

def innerTail142_9 : List Nat := 5563 :: innerTail142_10
def innerCandidates142_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates142_9_checked : candidates edge 4 829 5563 innerTail142_10 best.bound = innerCandidates142_9 := by
  unfold candidates
  have row : edge 5563 = SearchData.row5563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan142_9 : scan edge (search edge 3) 4 [829] 829 innerTail142_9 best = best := by
  rw [innerTail142_9, scan, if_neg (by decide), innerCandidates142_9_checked]
  change scan edge (search edge 3) 4 [829] 829 innerTail142_10
    (if 3 ≤ innerCandidates142_9.length then search edge 3 [829, 5563] innerCandidates142_9 6392 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan142_10

def innerTail142_8 : List Nat := 5527 :: innerTail142_9
def innerCandidates142_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates142_8_checked : candidates edge 4 829 5527 innerTail142_9 best.bound = innerCandidates142_8 := by
  unfold candidates
  have row : edge 5527 = SearchData.row5527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan142_8 : scan edge (search edge 3) 4 [829] 829 innerTail142_8 best = best := by
  rw [innerTail142_8, scan, if_neg (by decide), innerCandidates142_8_checked]
  change scan edge (search edge 3) 4 [829] 829 innerTail142_9
    (if 3 ≤ innerCandidates142_8.length then search edge 3 [829, 5527] innerCandidates142_8 6356 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan142_9

def innerTail142_7 : List Nat := 5107 :: innerTail142_8
def innerCandidates142_7 : List Nat := [5839]
set_option maxRecDepth 4096 in
theorem innerCandidates142_7_checked : candidates edge 4 829 5107 innerTail142_8 best.bound = innerCandidates142_7 := by
  unfold candidates
  have row : edge 5107 = SearchData.row5107.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan142_7 : scan edge (search edge 3) 4 [829] 829 innerTail142_7 best = best := by
  rw [innerTail142_7, scan, if_neg (by decide), innerCandidates142_7_checked]
  change scan edge (search edge 3) 4 [829] 829 innerTail142_8
    (if 3 ≤ innerCandidates142_7.length then search edge 3 [829, 5107] innerCandidates142_7 5936 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan142_8

def innerTail142_6 : List Nat := 3847 :: innerTail142_7
def innerCandidates142_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates142_6_checked : candidates edge 4 829 3847 innerTail142_7 best.bound = innerCandidates142_6 := by
  unfold candidates
  have row : edge 3847 = SearchData.row3847.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan142_6 : scan edge (search edge 3) 4 [829] 829 innerTail142_6 best = best := by
  rw [innerTail142_6, scan, if_neg (by decide), innerCandidates142_6_checked]
  change scan edge (search edge 3) 4 [829] 829 innerTail142_7
    (if 3 ≤ innerCandidates142_6.length then search edge 3 [829, 3847] innerCandidates142_6 4676 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan142_7

def innerTail142_5 : List Nat := 3361 :: innerTail142_6
def innerCandidates142_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates142_5_checked : candidates edge 4 829 3361 innerTail142_6 best.bound = innerCandidates142_5 := by
  unfold candidates
  have row : edge 3361 = SearchData.row3361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan142_5 : scan edge (search edge 3) 4 [829] 829 innerTail142_5 best = best := by
  rw [innerTail142_5, scan, if_neg (by decide), innerCandidates142_5_checked]
  change scan edge (search edge 3) 4 [829] 829 innerTail142_6
    (if 3 ≤ innerCandidates142_5.length then search edge 3 [829, 3361] innerCandidates142_5 4190 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan142_6

def innerTail142_4 : List Nat := 3319 :: innerTail142_5
def innerCandidates142_4 : List Nat := [7321]
set_option maxRecDepth 4096 in
theorem innerCandidates142_4_checked : candidates edge 4 829 3319 innerTail142_5 best.bound = innerCandidates142_4 := by
  unfold candidates
  have row : edge 3319 = SearchData.row3319.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan142_4 : scan edge (search edge 3) 4 [829] 829 innerTail142_4 best = best := by
  rw [innerTail142_4, scan, if_neg (by decide), innerCandidates142_4_checked]
  change scan edge (search edge 3) 4 [829] 829 innerTail142_5
    (if 3 ≤ innerCandidates142_4.length then search edge 3 [829, 3319] innerCandidates142_4 4148 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan142_5

def innerTail142_3 : List Nat := 2671 :: innerTail142_4
def innerCandidates142_3 : List Nat := [3361]
set_option maxRecDepth 4096 in
theorem innerCandidates142_3_checked : candidates edge 4 829 2671 innerTail142_4 best.bound = innerCandidates142_3 := by
  unfold candidates
  have row : edge 2671 = SearchData.row2671.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan142_3 : scan edge (search edge 3) 4 [829] 829 innerTail142_3 best = best := by
  rw [innerTail142_3, scan, if_neg (by decide), innerCandidates142_3_checked]
  change scan edge (search edge 3) 4 [829] 829 innerTail142_4
    (if 3 ≤ innerCandidates142_3.length then search edge 3 [829, 2671] innerCandidates142_3 3500 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan142_4

def innerTail142_2 : List Nat := 2293 :: innerTail142_3
def innerCandidates142_2 : List Nat := [2671]
set_option maxRecDepth 4096 in
theorem innerCandidates142_2_checked : candidates edge 4 829 2293 innerTail142_3 best.bound = innerCandidates142_2 := by
  unfold candidates
  have row : edge 2293 = PrimePairCache.smallRow2293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan142_2 : scan edge (search edge 3) 4 [829] 829 innerTail142_2 best = best := by
  rw [innerTail142_2, scan, if_neg (by decide), innerCandidates142_2_checked]
  change scan edge (search edge 3) 4 [829] 829 innerTail142_3
    (if 3 ≤ innerCandidates142_2.length then search edge 3 [829, 2293] innerCandidates142_2 3122 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan142_3

def innerTail142_1 : List Nat := 1237 :: innerTail142_2
def innerCandidates142_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates142_1_checked : candidates edge 4 829 1237 innerTail142_2 best.bound = innerCandidates142_1 := by
  unfold candidates
  have row : edge 1237 = PrimePairCache.smallRow1237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan142_1 : scan edge (search edge 3) 4 [829] 829 innerTail142_1 best = best := by
  rw [innerTail142_1, scan, if_neg (by decide), innerCandidates142_1_checked]
  change scan edge (search edge 3) 4 [829] 829 innerTail142_2
    (if 3 ≤ innerCandidates142_1.length then search edge 3 [829, 1237] innerCandidates142_1 2066 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan142_2

def innerTail142_0 : List Nat := 1033 :: innerTail142_1
def innerCandidates142_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates142_0_checked : candidates edge 4 829 1033 innerTail142_1 best.bound = innerCandidates142_0 := by
  unfold candidates
  have row : edge 1033 = PrimePairCache.smallRow1033.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan142_0 : scan edge (search edge 3) 4 [829] 829 innerTail142_0 best = best := by
  rw [innerTail142_0, scan, if_neg (by decide), innerCandidates142_0_checked]
  change scan edge (search edge 3) 4 [829] 829 innerTail142_1
    (if 3 ≤ innerCandidates142_0.length then search edge 3 [829, 1033] innerCandidates142_0 1862 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan142_1

theorem rootBranch142 : search edge 4 [829] rootCandidates142 829 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [829] 829 innerTail142_0 best = best
  exact innerScan142_0

end ProjectEulerSolutions.PrimePairUpper.Search
