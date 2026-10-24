import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates70 : List Nat := [457, 613, 751, 1021, 1117, 1531, 2293, 2521, 2659, 2791, 3217, 3463, 3529, 4339, 4597, 4663, 5059, 5563, 6091, 7207, 7459, 7573, 7789, 8317, 8377]
set_option maxRecDepth 4096 in
theorem rootCandidates70_checked : candidates edge 5 0 367 tail71 best.bound = rootCandidates70 := by
  unfold candidates
  have row : edge 367 = PrimePairCache.smallRow367.contains := by funext q; rfl
  rw [row]
  decide

def innerTail70_25 : List Nat := []
theorem innerScan70_25 : scan edge (search edge 3) 4 [367] 367 innerTail70_25 best = best := rfl

def innerTail70_24 : List Nat := 8377 :: innerTail70_25
def innerCandidates70_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates70_24_checked : candidates edge 4 367 8377 innerTail70_25 best.bound = innerCandidates70_24 := by
  unfold candidates
  have row : edge 8377 = SearchData.row8377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan70_24 : scan edge (search edge 3) 4 [367] 367 innerTail70_24 best = best := by
  rw [innerTail70_24, scan, if_neg (by decide), innerCandidates70_24_checked]
  change scan edge (search edge 3) 4 [367] 367 innerTail70_25
    (if 3 ≤ innerCandidates70_24.length then search edge 3 [367, 8377] innerCandidates70_24 8744 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan70_25

def innerTail70_23 : List Nat := 8317 :: innerTail70_24
def innerCandidates70_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates70_23_checked : candidates edge 4 367 8317 innerTail70_24 best.bound = innerCandidates70_23 := by
  unfold candidates
  have row : edge 8317 = SearchData.row8317.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan70_23 : scan edge (search edge 3) 4 [367] 367 innerTail70_23 best = best := by
  rw [innerTail70_23, scan, if_neg (by decide), innerCandidates70_23_checked]
  change scan edge (search edge 3) 4 [367] 367 innerTail70_24
    (if 3 ≤ innerCandidates70_23.length then search edge 3 [367, 8317] innerCandidates70_23 8684 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan70_24

def innerTail70_22 : List Nat := 7789 :: innerTail70_23
def innerCandidates70_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates70_22_checked : candidates edge 4 367 7789 innerTail70_23 best.bound = innerCandidates70_22 := by
  unfold candidates
  have row : edge 7789 = SearchData.row7789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan70_22 : scan edge (search edge 3) 4 [367] 367 innerTail70_22 best = best := by
  rw [innerTail70_22, scan, if_neg (by decide), innerCandidates70_22_checked]
  change scan edge (search edge 3) 4 [367] 367 innerTail70_23
    (if 3 ≤ innerCandidates70_22.length then search edge 3 [367, 7789] innerCandidates70_22 8156 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan70_23

def innerTail70_21 : List Nat := 7573 :: innerTail70_22
def innerCandidates70_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates70_21_checked : candidates edge 4 367 7573 innerTail70_22 best.bound = innerCandidates70_21 := by
  unfold candidates
  have row : edge 7573 = SearchData.row7573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan70_21 : scan edge (search edge 3) 4 [367] 367 innerTail70_21 best = best := by
  rw [innerTail70_21, scan, if_neg (by decide), innerCandidates70_21_checked]
  change scan edge (search edge 3) 4 [367] 367 innerTail70_22
    (if 3 ≤ innerCandidates70_21.length then search edge 3 [367, 7573] innerCandidates70_21 7940 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan70_22

def innerTail70_20 : List Nat := 7459 :: innerTail70_21
def innerCandidates70_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates70_20_checked : candidates edge 4 367 7459 innerTail70_21 best.bound = innerCandidates70_20 := by
  unfold candidates
  have row : edge 7459 = SearchData.row7459.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan70_20 : scan edge (search edge 3) 4 [367] 367 innerTail70_20 best = best := by
  rw [innerTail70_20, scan, if_neg (by decide), innerCandidates70_20_checked]
  change scan edge (search edge 3) 4 [367] 367 innerTail70_21
    (if 3 ≤ innerCandidates70_20.length then search edge 3 [367, 7459] innerCandidates70_20 7826 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan70_21

def innerTail70_19 : List Nat := 7207 :: innerTail70_20
def innerCandidates70_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates70_19_checked : candidates edge 4 367 7207 innerTail70_20 best.bound = innerCandidates70_19 := by
  unfold candidates
  have row : edge 7207 = SearchData.row7207.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan70_19 : scan edge (search edge 3) 4 [367] 367 innerTail70_19 best = best := by
  rw [innerTail70_19, scan, if_neg (by decide), innerCandidates70_19_checked]
  change scan edge (search edge 3) 4 [367] 367 innerTail70_20
    (if 3 ≤ innerCandidates70_19.length then search edge 3 [367, 7207] innerCandidates70_19 7574 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan70_20

def innerTail70_18 : List Nat := 6091 :: innerTail70_19
def innerCandidates70_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates70_18_checked : candidates edge 4 367 6091 innerTail70_19 best.bound = innerCandidates70_18 := by
  unfold candidates
  have row : edge 6091 = SearchData.row6091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan70_18 : scan edge (search edge 3) 4 [367] 367 innerTail70_18 best = best := by
  rw [innerTail70_18, scan, if_neg (by decide), innerCandidates70_18_checked]
  change scan edge (search edge 3) 4 [367] 367 innerTail70_19
    (if 3 ≤ innerCandidates70_18.length then search edge 3 [367, 6091] innerCandidates70_18 6458 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan70_19

def innerTail70_17 : List Nat := 5563 :: innerTail70_18
def innerCandidates70_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates70_17_checked : candidates edge 4 367 5563 innerTail70_18 best.bound = innerCandidates70_17 := by
  unfold candidates
  have row : edge 5563 = SearchData.row5563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan70_17 : scan edge (search edge 3) 4 [367] 367 innerTail70_17 best = best := by
  rw [innerTail70_17, scan, if_neg (by decide), innerCandidates70_17_checked]
  change scan edge (search edge 3) 4 [367] 367 innerTail70_18
    (if 3 ≤ innerCandidates70_17.length then search edge 3 [367, 5563] innerCandidates70_17 5930 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan70_18

def innerTail70_16 : List Nat := 5059 :: innerTail70_17
def innerCandidates70_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates70_16_checked : candidates edge 4 367 5059 innerTail70_17 best.bound = innerCandidates70_16 := by
  unfold candidates
  have row : edge 5059 = SearchData.row5059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan70_16 : scan edge (search edge 3) 4 [367] 367 innerTail70_16 best = best := by
  rw [innerTail70_16, scan, if_neg (by decide), innerCandidates70_16_checked]
  change scan edge (search edge 3) 4 [367] 367 innerTail70_17
    (if 3 ≤ innerCandidates70_16.length then search edge 3 [367, 5059] innerCandidates70_16 5426 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan70_17

def innerTail70_15 : List Nat := 4663 :: innerTail70_16
def innerCandidates70_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates70_15_checked : candidates edge 4 367 4663 innerTail70_16 best.bound = innerCandidates70_15 := by
  unfold candidates
  have row : edge 4663 = SearchData.row4663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan70_15 : scan edge (search edge 3) 4 [367] 367 innerTail70_15 best = best := by
  rw [innerTail70_15, scan, if_neg (by decide), innerCandidates70_15_checked]
  change scan edge (search edge 3) 4 [367] 367 innerTail70_16
    (if 3 ≤ innerCandidates70_15.length then search edge 3 [367, 4663] innerCandidates70_15 5030 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan70_16

def innerTail70_14 : List Nat := 4597 :: innerTail70_15
def innerCandidates70_14 : List Nat := [5563]
set_option maxRecDepth 4096 in
theorem innerCandidates70_14_checked : candidates edge 4 367 4597 innerTail70_15 best.bound = innerCandidates70_14 := by
  unfold candidates
  have row : edge 4597 = SearchData.row4597.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan70_14 : scan edge (search edge 3) 4 [367] 367 innerTail70_14 best = best := by
  rw [innerTail70_14, scan, if_neg (by decide), innerCandidates70_14_checked]
  change scan edge (search edge 3) 4 [367] 367 innerTail70_15
    (if 3 ≤ innerCandidates70_14.length then search edge 3 [367, 4597] innerCandidates70_14 4964 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan70_15

def innerTail70_13 : List Nat := 4339 :: innerTail70_14
def innerCandidates70_13 : List Nat := [8317]
set_option maxRecDepth 4096 in
theorem innerCandidates70_13_checked : candidates edge 4 367 4339 innerTail70_14 best.bound = innerCandidates70_13 := by
  unfold candidates
  have row : edge 4339 = SearchData.row4339.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan70_13 : scan edge (search edge 3) 4 [367] 367 innerTail70_13 best = best := by
  rw [innerTail70_13, scan, if_neg (by decide), innerCandidates70_13_checked]
  change scan edge (search edge 3) 4 [367] 367 innerTail70_14
    (if 3 ≤ innerCandidates70_13.length then search edge 3 [367, 4339] innerCandidates70_13 4706 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan70_14

def innerTail70_12 : List Nat := 3529 :: innerTail70_13
def innerCandidates70_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates70_12_checked : candidates edge 4 367 3529 innerTail70_13 best.bound = innerCandidates70_12 := by
  unfold candidates
  have row : edge 3529 = SearchData.row3529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan70_12 : scan edge (search edge 3) 4 [367] 367 innerTail70_12 best = best := by
  rw [innerTail70_12, scan, if_neg (by decide), innerCandidates70_12_checked]
  change scan edge (search edge 3) 4 [367] 367 innerTail70_13
    (if 3 ≤ innerCandidates70_12.length then search edge 3 [367, 3529] innerCandidates70_12 3896 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan70_13

def innerTail70_11 : List Nat := 3463 :: innerTail70_12
def innerCandidates70_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates70_11_checked : candidates edge 4 367 3463 innerTail70_12 best.bound = innerCandidates70_11 := by
  unfold candidates
  have row : edge 3463 = SearchData.row3463.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan70_11 : scan edge (search edge 3) 4 [367] 367 innerTail70_11 best = best := by
  rw [innerTail70_11, scan, if_neg (by decide), innerCandidates70_11_checked]
  change scan edge (search edge 3) 4 [367] 367 innerTail70_12
    (if 3 ≤ innerCandidates70_11.length then search edge 3 [367, 3463] innerCandidates70_11 3830 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan70_12

def innerTail70_10 : List Nat := 3217 :: innerTail70_11
def innerCandidates70_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates70_10_checked : candidates edge 4 367 3217 innerTail70_11 best.bound = innerCandidates70_10 := by
  unfold candidates
  have row : edge 3217 = SearchData.row3217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan70_10 : scan edge (search edge 3) 4 [367] 367 innerTail70_10 best = best := by
  rw [innerTail70_10, scan, if_neg (by decide), innerCandidates70_10_checked]
  change scan edge (search edge 3) 4 [367] 367 innerTail70_11
    (if 3 ≤ innerCandidates70_10.length then search edge 3 [367, 3217] innerCandidates70_10 3584 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan70_11

def innerTail70_9 : List Nat := 2791 :: innerTail70_10
def innerCandidates70_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates70_9_checked : candidates edge 4 367 2791 innerTail70_10 best.bound = innerCandidates70_9 := by
  unfold candidates
  have row : edge 2791 = SearchData.row2791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan70_9 : scan edge (search edge 3) 4 [367] 367 innerTail70_9 best = best := by
  rw [innerTail70_9, scan, if_neg (by decide), innerCandidates70_9_checked]
  change scan edge (search edge 3) 4 [367] 367 innerTail70_10
    (if 3 ≤ innerCandidates70_9.length then search edge 3 [367, 2791] innerCandidates70_9 3158 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan70_10

def innerTail70_8 : List Nat := 2659 :: innerTail70_9
def innerCandidates70_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates70_8_checked : candidates edge 4 367 2659 innerTail70_9 best.bound = innerCandidates70_8 := by
  unfold candidates
  have row : edge 2659 = PrimePairCache.smallRow2659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan70_8 : scan edge (search edge 3) 4 [367] 367 innerTail70_8 best = best := by
  rw [innerTail70_8, scan, if_neg (by decide), innerCandidates70_8_checked]
  change scan edge (search edge 3) 4 [367] 367 innerTail70_9
    (if 3 ≤ innerCandidates70_8.length then search edge 3 [367, 2659] innerCandidates70_8 3026 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan70_9

def innerTail70_7 : List Nat := 2521 :: innerTail70_8
def innerCandidates70_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates70_7_checked : candidates edge 4 367 2521 innerTail70_8 best.bound = innerCandidates70_7 := by
  unfold candidates
  have row : edge 2521 = PrimePairCache.smallRow2521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan70_7 : scan edge (search edge 3) 4 [367] 367 innerTail70_7 best = best := by
  rw [innerTail70_7, scan, if_neg (by decide), innerCandidates70_7_checked]
  change scan edge (search edge 3) 4 [367] 367 innerTail70_8
    (if 3 ≤ innerCandidates70_7.length then search edge 3 [367, 2521] innerCandidates70_7 2888 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan70_8

def innerTail70_6 : List Nat := 2293 :: innerTail70_7
def innerCandidates70_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates70_6_checked : candidates edge 4 367 2293 innerTail70_7 best.bound = innerCandidates70_6 := by
  unfold candidates
  have row : edge 2293 = PrimePairCache.smallRow2293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan70_6 : scan edge (search edge 3) 4 [367] 367 innerTail70_6 best = best := by
  rw [innerTail70_6, scan, if_neg (by decide), innerCandidates70_6_checked]
  change scan edge (search edge 3) 4 [367] 367 innerTail70_7
    (if 3 ≤ innerCandidates70_6.length then search edge 3 [367, 2293] innerCandidates70_6 2660 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan70_7

def innerTail70_5 : List Nat := 1531 :: innerTail70_6
def innerCandidates70_5 : List Nat := [5059, 7789]
set_option maxRecDepth 4096 in
theorem innerCandidates70_5_checked : candidates edge 4 367 1531 innerTail70_6 best.bound = innerCandidates70_5 := by
  unfold candidates
  have row : edge 1531 = PrimePairCache.smallRow1531.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan70_5 : scan edge (search edge 3) 4 [367] 367 innerTail70_5 best = best := by
  rw [innerTail70_5, scan, if_neg (by decide), innerCandidates70_5_checked]
  change scan edge (search edge 3) 4 [367] 367 innerTail70_6
    (if 3 ≤ innerCandidates70_5.length then search edge 3 [367, 1531] innerCandidates70_5 1898 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan70_6

def innerTail70_4 : List Nat := 1117 :: innerTail70_5
def innerCandidates70_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates70_4_checked : candidates edge 4 367 1117 innerTail70_5 best.bound = innerCandidates70_4 := by
  unfold candidates
  have row : edge 1117 = PrimePairCache.smallRow1117.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan70_4 : scan edge (search edge 3) 4 [367] 367 innerTail70_4 best = best := by
  rw [innerTail70_4, scan, if_neg (by decide), innerCandidates70_4_checked]
  change scan edge (search edge 3) 4 [367] 367 innerTail70_5
    (if 3 ≤ innerCandidates70_4.length then search edge 3 [367, 1117] innerCandidates70_4 1484 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan70_5

def innerTail70_3 : List Nat := 1021 :: innerTail70_4
def innerCandidates70_3 : List Nat := [3529, 4597]
set_option maxRecDepth 4096 in
theorem innerCandidates70_3_checked : candidates edge 4 367 1021 innerTail70_4 best.bound = innerCandidates70_3 := by
  unfold candidates
  have row : edge 1021 = PrimePairCache.smallRow1021.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan70_3 : scan edge (search edge 3) 4 [367] 367 innerTail70_3 best = best := by
  rw [innerTail70_3, scan, if_neg (by decide), innerCandidates70_3_checked]
  change scan edge (search edge 3) 4 [367] 367 innerTail70_4
    (if 3 ≤ innerCandidates70_3.length then search edge 3 [367, 1021] innerCandidates70_3 1388 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan70_4

def innerTail70_2 : List Nat := 751 :: innerTail70_3
def innerCandidates70_2 : List Nat := [4597]
set_option maxRecDepth 4096 in
theorem innerCandidates70_2_checked : candidates edge 4 367 751 innerTail70_3 best.bound = innerCandidates70_2 := by
  unfold candidates
  have row : edge 751 = PrimePairCache.smallRow751.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan70_2 : scan edge (search edge 3) 4 [367] 367 innerTail70_2 best = best := by
  rw [innerTail70_2, scan, if_neg (by decide), innerCandidates70_2_checked]
  change scan edge (search edge 3) 4 [367] 367 innerTail70_3
    (if 3 ≤ innerCandidates70_2.length then search edge 3 [367, 751] innerCandidates70_2 1118 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan70_3

def innerTail70_1 : List Nat := 613 :: innerTail70_2
def innerCandidates70_1 : List Nat := [4339]
set_option maxRecDepth 4096 in
theorem innerCandidates70_1_checked : candidates edge 4 367 613 innerTail70_2 best.bound = innerCandidates70_1 := by
  unfold candidates
  have row : edge 613 = PrimePairCache.smallRow613.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan70_1 : scan edge (search edge 3) 4 [367] 367 innerTail70_1 best = best := by
  rw [innerTail70_1, scan, if_neg (by decide), innerCandidates70_1_checked]
  change scan edge (search edge 3) 4 [367] 367 innerTail70_2
    (if 3 ≤ innerCandidates70_1.length then search edge 3 [367, 613] innerCandidates70_1 980 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan70_2

def innerTail70_0 : List Nat := 457 :: innerTail70_1
def innerCandidates70_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates70_0_checked : candidates edge 4 367 457 innerTail70_1 best.bound = innerCandidates70_0 := by
  unfold candidates
  have row : edge 457 = PrimePairCache.smallRow457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan70_0 : scan edge (search edge 3) 4 [367] 367 innerTail70_0 best = best := by
  rw [innerTail70_0, scan, if_neg (by decide), innerCandidates70_0_checked]
  change scan edge (search edge 3) 4 [367] 367 innerTail70_1
    (if 3 ≤ innerCandidates70_0.length then search edge 3 [367, 457] innerCandidates70_0 824 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan70_1

theorem rootBranch70 : search edge 4 [367] rootCandidates70 367 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [367] 367 innerTail70_0 best = best
  exact innerScan70_0

def rootCandidates69 : List Nat := [509, 563, 599, 701, 911, 1487, 1499, 1667, 1949, 2243, 2399, 2447, 2969, 3203, 3539, 3833, 3881, 5591, 6143, 6689, 7001, 7907, 8039]
set_option maxRecDepth 4096 in
theorem rootCandidates69_checked : candidates edge 5 0 359 tail70 best.bound = rootCandidates69 := by
  unfold candidates
  have row : edge 359 = PrimePairCache.smallRow359.contains := by funext q; rfl
  rw [row]
  decide

def innerTail69_23 : List Nat := []
theorem innerScan69_23 : scan edge (search edge 3) 4 [359] 359 innerTail69_23 best = best := rfl

def innerTail69_22 : List Nat := 8039 :: innerTail69_23
def innerCandidates69_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates69_22_checked : candidates edge 4 359 8039 innerTail69_23 best.bound = innerCandidates69_22 := by
  unfold candidates
  have row : edge 8039 = SearchData.row8039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan69_22 : scan edge (search edge 3) 4 [359] 359 innerTail69_22 best = best := by
  rw [innerTail69_22, scan, if_neg (by decide), innerCandidates69_22_checked]
  change scan edge (search edge 3) 4 [359] 359 innerTail69_23
    (if 3 ≤ innerCandidates69_22.length then search edge 3 [359, 8039] innerCandidates69_22 8398 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan69_23

def innerTail69_21 : List Nat := 7907 :: innerTail69_22
def innerCandidates69_21 : List Nat := [8039]
set_option maxRecDepth 4096 in
theorem innerCandidates69_21_checked : candidates edge 4 359 7907 innerTail69_22 best.bound = innerCandidates69_21 := by
  unfold candidates
  have row : edge 7907 = SearchData.row7907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan69_21 : scan edge (search edge 3) 4 [359] 359 innerTail69_21 best = best := by
  rw [innerTail69_21, scan, if_neg (by decide), innerCandidates69_21_checked]
  change scan edge (search edge 3) 4 [359] 359 innerTail69_22
    (if 3 ≤ innerCandidates69_21.length then search edge 3 [359, 7907] innerCandidates69_21 8266 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan69_22

def innerTail69_20 : List Nat := 7001 :: innerTail69_21
def innerCandidates69_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates69_20_checked : candidates edge 4 359 7001 innerTail69_21 best.bound = innerCandidates69_20 := by
  unfold candidates
  have row : edge 7001 = SearchData.row7001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan69_20 : scan edge (search edge 3) 4 [359] 359 innerTail69_20 best = best := by
  rw [innerTail69_20, scan, if_neg (by decide), innerCandidates69_20_checked]
  change scan edge (search edge 3) 4 [359] 359 innerTail69_21
    (if 3 ≤ innerCandidates69_20.length then search edge 3 [359, 7001] innerCandidates69_20 7360 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan69_21

def innerTail69_19 : List Nat := 6689 :: innerTail69_20
def innerCandidates69_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates69_19_checked : candidates edge 4 359 6689 innerTail69_20 best.bound = innerCandidates69_19 := by
  unfold candidates
  have row : edge 6689 = SearchData.row6689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan69_19 : scan edge (search edge 3) 4 [359] 359 innerTail69_19 best = best := by
  rw [innerTail69_19, scan, if_neg (by decide), innerCandidates69_19_checked]
  change scan edge (search edge 3) 4 [359] 359 innerTail69_20
    (if 3 ≤ innerCandidates69_19.length then search edge 3 [359, 6689] innerCandidates69_19 7048 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan69_20

def innerTail69_18 : List Nat := 6143 :: innerTail69_19
def innerCandidates69_18 : List Nat := [6689]
set_option maxRecDepth 4096 in
theorem innerCandidates69_18_checked : candidates edge 4 359 6143 innerTail69_19 best.bound = innerCandidates69_18 := by
  unfold candidates
  have row : edge 6143 = SearchData.row6143.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan69_18 : scan edge (search edge 3) 4 [359] 359 innerTail69_18 best = best := by
  rw [innerTail69_18, scan, if_neg (by decide), innerCandidates69_18_checked]
  change scan edge (search edge 3) 4 [359] 359 innerTail69_19
    (if 3 ≤ innerCandidates69_18.length then search edge 3 [359, 6143] innerCandidates69_18 6502 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan69_19

def innerTail69_17 : List Nat := 5591 :: innerTail69_18
def innerCandidates69_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates69_17_checked : candidates edge 4 359 5591 innerTail69_18 best.bound = innerCandidates69_17 := by
  unfold candidates
  have row : edge 5591 = SearchData.row5591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan69_17 : scan edge (search edge 3) 4 [359] 359 innerTail69_17 best = best := by
  rw [innerTail69_17, scan, if_neg (by decide), innerCandidates69_17_checked]
  change scan edge (search edge 3) 4 [359] 359 innerTail69_18
    (if 3 ≤ innerCandidates69_17.length then search edge 3 [359, 5591] innerCandidates69_17 5950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan69_18

def innerTail69_16 : List Nat := 3881 :: innerTail69_17
def innerCandidates69_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates69_16_checked : candidates edge 4 359 3881 innerTail69_17 best.bound = innerCandidates69_16 := by
  unfold candidates
  have row : edge 3881 = SearchData.row3881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan69_16 : scan edge (search edge 3) 4 [359] 359 innerTail69_16 best = best := by
  rw [innerTail69_16, scan, if_neg (by decide), innerCandidates69_16_checked]
  change scan edge (search edge 3) 4 [359] 359 innerTail69_17
    (if 3 ≤ innerCandidates69_16.length then search edge 3 [359, 3881] innerCandidates69_16 4240 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan69_17

def innerTail69_15 : List Nat := 3833 :: innerTail69_16
def innerCandidates69_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates69_15_checked : candidates edge 4 359 3833 innerTail69_16 best.bound = innerCandidates69_15 := by
  unfold candidates
  have row : edge 3833 = SearchData.row3833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan69_15 : scan edge (search edge 3) 4 [359] 359 innerTail69_15 best = best := by
  rw [innerTail69_15, scan, if_neg (by decide), innerCandidates69_15_checked]
  change scan edge (search edge 3) 4 [359] 359 innerTail69_16
    (if 3 ≤ innerCandidates69_15.length then search edge 3 [359, 3833] innerCandidates69_15 4192 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan69_16

def innerTail69_14 : List Nat := 3539 :: innerTail69_15
def innerCandidates69_14 : List Nat := [7001]
set_option maxRecDepth 4096 in
theorem innerCandidates69_14_checked : candidates edge 4 359 3539 innerTail69_15 best.bound = innerCandidates69_14 := by
  unfold candidates
  have row : edge 3539 = SearchData.row3539.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan69_14 : scan edge (search edge 3) 4 [359] 359 innerTail69_14 best = best := by
  rw [innerTail69_14, scan, if_neg (by decide), innerCandidates69_14_checked]
  change scan edge (search edge 3) 4 [359] 359 innerTail69_15
    (if 3 ≤ innerCandidates69_14.length then search edge 3 [359, 3539] innerCandidates69_14 3898 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan69_15

def innerTail69_13 : List Nat := 3203 :: innerTail69_14
def innerCandidates69_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates69_13_checked : candidates edge 4 359 3203 innerTail69_14 best.bound = innerCandidates69_13 := by
  unfold candidates
  have row : edge 3203 = SearchData.row3203.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan69_13 : scan edge (search edge 3) 4 [359] 359 innerTail69_13 best = best := by
  rw [innerTail69_13, scan, if_neg (by decide), innerCandidates69_13_checked]
  change scan edge (search edge 3) 4 [359] 359 innerTail69_14
    (if 3 ≤ innerCandidates69_13.length then search edge 3 [359, 3203] innerCandidates69_13 3562 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan69_14

def innerTail69_12 : List Nat := 2969 :: innerTail69_13
def innerCandidates69_12 : List Nat := [3539]
set_option maxRecDepth 4096 in
theorem innerCandidates69_12_checked : candidates edge 4 359 2969 innerTail69_13 best.bound = innerCandidates69_12 := by
  unfold candidates
  have row : edge 2969 = SearchData.row2969.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan69_12 : scan edge (search edge 3) 4 [359] 359 innerTail69_12 best = best := by
  rw [innerTail69_12, scan, if_neg (by decide), innerCandidates69_12_checked]
  change scan edge (search edge 3) 4 [359] 359 innerTail69_13
    (if 3 ≤ innerCandidates69_12.length then search edge 3 [359, 2969] innerCandidates69_12 3328 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan69_13

def innerTail69_11 : List Nat := 2447 :: innerTail69_12
def innerCandidates69_11 : List Nat := [7001, 7907]
set_option maxRecDepth 4096 in
theorem innerCandidates69_11_checked : candidates edge 4 359 2447 innerTail69_12 best.bound = innerCandidates69_11 := by
  unfold candidates
  have row : edge 2447 = PrimePairCache.smallRow2447.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan69_11 : scan edge (search edge 3) 4 [359] 359 innerTail69_11 best = best := by
  rw [innerTail69_11, scan, if_neg (by decide), innerCandidates69_11_checked]
  change scan edge (search edge 3) 4 [359] 359 innerTail69_12
    (if 3 ≤ innerCandidates69_11.length then search edge 3 [359, 2447] innerCandidates69_11 2806 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan69_12

def innerTail69_10 : List Nat := 2399 :: innerTail69_11
def innerCandidates69_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates69_10_checked : candidates edge 4 359 2399 innerTail69_11 best.bound = innerCandidates69_10 := by
  unfold candidates
  have row : edge 2399 = PrimePairCache.smallRow2399.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan69_10 : scan edge (search edge 3) 4 [359] 359 innerTail69_10 best = best := by
  rw [innerTail69_10, scan, if_neg (by decide), innerCandidates69_10_checked]
  change scan edge (search edge 3) 4 [359] 359 innerTail69_11
    (if 3 ≤ innerCandidates69_10.length then search edge 3 [359, 2399] innerCandidates69_10 2758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan69_11

def innerTail69_9 : List Nat := 2243 :: innerTail69_10
def innerCandidates69_9 : List Nat := [3539]
set_option maxRecDepth 4096 in
theorem innerCandidates69_9_checked : candidates edge 4 359 2243 innerTail69_10 best.bound = innerCandidates69_9 := by
  unfold candidates
  have row : edge 2243 = PrimePairCache.smallRow2243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan69_9 : scan edge (search edge 3) 4 [359] 359 innerTail69_9 best = best := by
  rw [innerTail69_9, scan, if_neg (by decide), innerCandidates69_9_checked]
  change scan edge (search edge 3) 4 [359] 359 innerTail69_10
    (if 3 ≤ innerCandidates69_9.length then search edge 3 [359, 2243] innerCandidates69_9 2602 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan69_10

def innerTail69_8 : List Nat := 1949 :: innerTail69_9
def innerCandidates69_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates69_8_checked : candidates edge 4 359 1949 innerTail69_9 best.bound = innerCandidates69_8 := by
  unfold candidates
  have row : edge 1949 = PrimePairCache.smallRow1949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan69_8 : scan edge (search edge 3) 4 [359] 359 innerTail69_8 best = best := by
  rw [innerTail69_8, scan, if_neg (by decide), innerCandidates69_8_checked]
  change scan edge (search edge 3) 4 [359] 359 innerTail69_9
    (if 3 ≤ innerCandidates69_8.length then search edge 3 [359, 1949] innerCandidates69_8 2308 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan69_9

def innerTail69_7 : List Nat := 1667 :: innerTail69_8
def innerCandidates69_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates69_7_checked : candidates edge 4 359 1667 innerTail69_8 best.bound = innerCandidates69_7 := by
  unfold candidates
  have row : edge 1667 = PrimePairCache.smallRow1667.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan69_7 : scan edge (search edge 3) 4 [359] 359 innerTail69_7 best = best := by
  rw [innerTail69_7, scan, if_neg (by decide), innerCandidates69_7_checked]
  change scan edge (search edge 3) 4 [359] 359 innerTail69_8
    (if 3 ≤ innerCandidates69_7.length then search edge 3 [359, 1667] innerCandidates69_7 2026 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan69_8

def innerTail69_6 : List Nat := 1499 :: innerTail69_7
def innerCandidates69_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates69_6_checked : candidates edge 4 359 1499 innerTail69_7 best.bound = innerCandidates69_6 := by
  unfold candidates
  have row : edge 1499 = PrimePairCache.smallRow1499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan69_6 : scan edge (search edge 3) 4 [359] 359 innerTail69_6 best = best := by
  rw [innerTail69_6, scan, if_neg (by decide), innerCandidates69_6_checked]
  change scan edge (search edge 3) 4 [359] 359 innerTail69_7
    (if 3 ≤ innerCandidates69_6.length then search edge 3 [359, 1499] innerCandidates69_6 1858 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan69_7

def innerTail69_5 : List Nat := 1487 :: innerTail69_6
def innerCandidates69_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates69_5_checked : candidates edge 4 359 1487 innerTail69_6 best.bound = innerCandidates69_5 := by
  unfold candidates
  have row : edge 1487 = PrimePairCache.smallRow1487.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan69_5 : scan edge (search edge 3) 4 [359] 359 innerTail69_5 best = best := by
  rw [innerTail69_5, scan, if_neg (by decide), innerCandidates69_5_checked]
  change scan edge (search edge 3) 4 [359] 359 innerTail69_6
    (if 3 ≤ innerCandidates69_5.length then search edge 3 [359, 1487] innerCandidates69_5 1846 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan69_6

def innerTail69_4 : List Nat := 911 :: innerTail69_5
def innerCandidates69_4 : List Nat := [2399]
set_option maxRecDepth 4096 in
theorem innerCandidates69_4_checked : candidates edge 4 359 911 innerTail69_5 best.bound = innerCandidates69_4 := by
  unfold candidates
  have row : edge 911 = PrimePairCache.smallRow911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan69_4 : scan edge (search edge 3) 4 [359] 359 innerTail69_4 best = best := by
  rw [innerTail69_4, scan, if_neg (by decide), innerCandidates69_4_checked]
  change scan edge (search edge 3) 4 [359] 359 innerTail69_5
    (if 3 ≤ innerCandidates69_4.length then search edge 3 [359, 911] innerCandidates69_4 1270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan69_5

def innerTail69_3 : List Nat := 701 :: innerTail69_4
def innerCandidates69_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates69_3_checked : candidates edge 4 359 701 innerTail69_4 best.bound = innerCandidates69_3 := by
  unfold candidates
  have row : edge 701 = PrimePairCache.smallRow701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan69_3 : scan edge (search edge 3) 4 [359] 359 innerTail69_3 best = best := by
  rw [innerTail69_3, scan, if_neg (by decide), innerCandidates69_3_checked]
  change scan edge (search edge 3) 4 [359] 359 innerTail69_4
    (if 3 ≤ innerCandidates69_3.length then search edge 3 [359, 701] innerCandidates69_3 1060 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan69_4

def innerTail69_2 : List Nat := 599 :: innerTail69_3
def innerCandidates69_2 : List Nat := [8039]
set_option maxRecDepth 4096 in
theorem innerCandidates69_2_checked : candidates edge 4 359 599 innerTail69_3 best.bound = innerCandidates69_2 := by
  unfold candidates
  have row : edge 599 = PrimePairCache.smallRow599.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan69_2 : scan edge (search edge 3) 4 [359] 359 innerTail69_2 best = best := by
  rw [innerTail69_2, scan, if_neg (by decide), innerCandidates69_2_checked]
  change scan edge (search edge 3) 4 [359] 359 innerTail69_3
    (if 3 ≤ innerCandidates69_2.length then search edge 3 [359, 599] innerCandidates69_2 958 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan69_3

def innerTail69_1 : List Nat := 563 :: innerTail69_2
def innerCandidates69_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates69_1_checked : candidates edge 4 359 563 innerTail69_2 best.bound = innerCandidates69_1 := by
  unfold candidates
  have row : edge 563 = PrimePairCache.smallRow563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan69_1 : scan edge (search edge 3) 4 [359] 359 innerTail69_1 best = best := by
  rw [innerTail69_1, scan, if_neg (by decide), innerCandidates69_1_checked]
  change scan edge (search edge 3) 4 [359] 359 innerTail69_2
    (if 3 ≤ innerCandidates69_1.length then search edge 3 [359, 563] innerCandidates69_1 922 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan69_2

def innerTail69_0 : List Nat := 509 :: innerTail69_1
def innerCandidates69_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates69_0_checked : candidates edge 4 359 509 innerTail69_1 best.bound = innerCandidates69_0 := by
  unfold candidates
  have row : edge 509 = PrimePairCache.smallRow509.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan69_0 : scan edge (search edge 3) 4 [359] 359 innerTail69_0 best = best := by
  rw [innerTail69_0, scan, if_neg (by decide), innerCandidates69_0_checked]
  change scan edge (search edge 3) 4 [359] 359 innerTail69_1
    (if 3 ≤ innerCandidates69_0.length then search edge 3 [359, 509] innerCandidates69_0 868 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan69_1

theorem rootBranch69 : search edge 4 [359] rootCandidates69 359 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [359] 359 innerTail69_0 best = best
  exact innerScan69_0

end ProjectEulerSolutions.PrimePairUpper.Search
