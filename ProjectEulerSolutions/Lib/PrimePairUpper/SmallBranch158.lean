import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates201 : List Nat := [1867, 2131, 2377, 2551, 3391, 3727, 4441, 5563, 5569, 5851, 6079, 6151, 6871, 6949, 7369, 7621]
set_option maxRecDepth 4096 in
theorem rootCandidates201_checked : candidates edge 5 0 1249 tail202 best.bound = rootCandidates201 := by
  unfold candidates
  have row : edge 1249 = PrimePairCache.smallRow1249.contains := by funext q; rfl
  rw [row]
  decide

def innerTail201_16 : List Nat := []
theorem innerScan201_16 : scan edge (search edge 3) 4 [1249] 1249 innerTail201_16 best = best := rfl

def innerTail201_15 : List Nat := 7621 :: innerTail201_16
def innerCandidates201_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates201_15_checked : candidates edge 4 1249 7621 innerTail201_16 best.bound = innerCandidates201_15 := by
  unfold candidates
  have row : edge 7621 = SearchData.row7621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan201_15 : scan edge (search edge 3) 4 [1249] 1249 innerTail201_15 best = best := by
  rw [innerTail201_15, scan, if_neg (by decide), innerCandidates201_15_checked]
  change scan edge (search edge 3) 4 [1249] 1249 innerTail201_16
    (if 3 ≤ innerCandidates201_15.length then search edge 3 [1249, 7621] innerCandidates201_15 8870 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan201_16

def innerTail201_14 : List Nat := 7369 :: innerTail201_15
def innerCandidates201_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates201_14_checked : candidates edge 4 1249 7369 innerTail201_15 best.bound = innerCandidates201_14 := by
  unfold candidates
  have row : edge 7369 = SearchData.row7369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan201_14 : scan edge (search edge 3) 4 [1249] 1249 innerTail201_14 best = best := by
  rw [innerTail201_14, scan, if_neg (by decide), innerCandidates201_14_checked]
  change scan edge (search edge 3) 4 [1249] 1249 innerTail201_15
    (if 3 ≤ innerCandidates201_14.length then search edge 3 [1249, 7369] innerCandidates201_14 8618 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan201_15

def innerTail201_13 : List Nat := 6949 :: innerTail201_14
def innerCandidates201_13 : List Nat := [7621]
set_option maxRecDepth 4096 in
theorem innerCandidates201_13_checked : candidates edge 4 1249 6949 innerTail201_14 best.bound = innerCandidates201_13 := by
  unfold candidates
  have row : edge 6949 = SearchData.row6949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan201_13 : scan edge (search edge 3) 4 [1249] 1249 innerTail201_13 best = best := by
  rw [innerTail201_13, scan, if_neg (by decide), innerCandidates201_13_checked]
  change scan edge (search edge 3) 4 [1249] 1249 innerTail201_14
    (if 3 ≤ innerCandidates201_13.length then search edge 3 [1249, 6949] innerCandidates201_13 8198 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan201_14

def innerTail201_12 : List Nat := 6871 :: innerTail201_13
def innerCandidates201_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates201_12_checked : candidates edge 4 1249 6871 innerTail201_13 best.bound = innerCandidates201_12 := by
  unfold candidates
  have row : edge 6871 = SearchData.row6871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan201_12 : scan edge (search edge 3) 4 [1249] 1249 innerTail201_12 best = best := by
  rw [innerTail201_12, scan, if_neg (by decide), innerCandidates201_12_checked]
  change scan edge (search edge 3) 4 [1249] 1249 innerTail201_13
    (if 3 ≤ innerCandidates201_12.length then search edge 3 [1249, 6871] innerCandidates201_12 8120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan201_13

def innerTail201_11 : List Nat := 6151 :: innerTail201_12
def innerCandidates201_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates201_11_checked : candidates edge 4 1249 6151 innerTail201_12 best.bound = innerCandidates201_11 := by
  unfold candidates
  have row : edge 6151 = SearchData.row6151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan201_11 : scan edge (search edge 3) 4 [1249] 1249 innerTail201_11 best = best := by
  rw [innerTail201_11, scan, if_neg (by decide), innerCandidates201_11_checked]
  change scan edge (search edge 3) 4 [1249] 1249 innerTail201_12
    (if 3 ≤ innerCandidates201_11.length then search edge 3 [1249, 6151] innerCandidates201_11 7400 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan201_12

def innerTail201_10 : List Nat := 6079 :: innerTail201_11
def innerCandidates201_10 : List Nat := [7621]
set_option maxRecDepth 4096 in
theorem innerCandidates201_10_checked : candidates edge 4 1249 6079 innerTail201_11 best.bound = innerCandidates201_10 := by
  unfold candidates
  have row : edge 6079 = SearchData.row6079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan201_10 : scan edge (search edge 3) 4 [1249] 1249 innerTail201_10 best = best := by
  rw [innerTail201_10, scan, if_neg (by decide), innerCandidates201_10_checked]
  change scan edge (search edge 3) 4 [1249] 1249 innerTail201_11
    (if 3 ≤ innerCandidates201_10.length then search edge 3 [1249, 6079] innerCandidates201_10 7328 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan201_11

def innerTail201_9 : List Nat := 5851 :: innerTail201_10
def innerCandidates201_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates201_9_checked : candidates edge 4 1249 5851 innerTail201_10 best.bound = innerCandidates201_9 := by
  unfold candidates
  have row : edge 5851 = SearchData.row5851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan201_9 : scan edge (search edge 3) 4 [1249] 1249 innerTail201_9 best = best := by
  rw [innerTail201_9, scan, if_neg (by decide), innerCandidates201_9_checked]
  change scan edge (search edge 3) 4 [1249] 1249 innerTail201_10
    (if 3 ≤ innerCandidates201_9.length then search edge 3 [1249, 5851] innerCandidates201_9 7100 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan201_10

def innerTail201_8 : List Nat := 5569 :: innerTail201_9
def innerCandidates201_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates201_8_checked : candidates edge 4 1249 5569 innerTail201_9 best.bound = innerCandidates201_8 := by
  unfold candidates
  have row : edge 5569 = SearchData.row5569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan201_8 : scan edge (search edge 3) 4 [1249] 1249 innerTail201_8 best = best := by
  rw [innerTail201_8, scan, if_neg (by decide), innerCandidates201_8_checked]
  change scan edge (search edge 3) 4 [1249] 1249 innerTail201_9
    (if 3 ≤ innerCandidates201_8.length then search edge 3 [1249, 5569] innerCandidates201_8 6818 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan201_9

def innerTail201_7 : List Nat := 5563 :: innerTail201_8
def innerCandidates201_7 : List Nat := [6079]
set_option maxRecDepth 4096 in
theorem innerCandidates201_7_checked : candidates edge 4 1249 5563 innerTail201_8 best.bound = innerCandidates201_7 := by
  unfold candidates
  have row : edge 5563 = SearchData.row5563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan201_7 : scan edge (search edge 3) 4 [1249] 1249 innerTail201_7 best = best := by
  rw [innerTail201_7, scan, if_neg (by decide), innerCandidates201_7_checked]
  change scan edge (search edge 3) 4 [1249] 1249 innerTail201_8
    (if 3 ≤ innerCandidates201_7.length then search edge 3 [1249, 5563] innerCandidates201_7 6812 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan201_8

def innerTail201_6 : List Nat := 4441 :: innerTail201_7
def innerCandidates201_6 : List Nat := [6949]
set_option maxRecDepth 4096 in
theorem innerCandidates201_6_checked : candidates edge 4 1249 4441 innerTail201_7 best.bound = innerCandidates201_6 := by
  unfold candidates
  have row : edge 4441 = SearchData.row4441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan201_6 : scan edge (search edge 3) 4 [1249] 1249 innerTail201_6 best = best := by
  rw [innerTail201_6, scan, if_neg (by decide), innerCandidates201_6_checked]
  change scan edge (search edge 3) 4 [1249] 1249 innerTail201_7
    (if 3 ≤ innerCandidates201_6.length then search edge 3 [1249, 4441] innerCandidates201_6 5690 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan201_7

def innerTail201_5 : List Nat := 3727 :: innerTail201_6
def innerCandidates201_5 : List Nat := [6949]
set_option maxRecDepth 4096 in
theorem innerCandidates201_5_checked : candidates edge 4 1249 3727 innerTail201_6 best.bound = innerCandidates201_5 := by
  unfold candidates
  have row : edge 3727 = SearchData.row3727.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan201_5 : scan edge (search edge 3) 4 [1249] 1249 innerTail201_5 best = best := by
  rw [innerTail201_5, scan, if_neg (by decide), innerCandidates201_5_checked]
  change scan edge (search edge 3) 4 [1249] 1249 innerTail201_6
    (if 3 ≤ innerCandidates201_5.length then search edge 3 [1249, 3727] innerCandidates201_5 4976 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan201_6

def innerTail201_4 : List Nat := 3391 :: innerTail201_5
def innerCandidates201_4 : List Nat := [6079]
set_option maxRecDepth 4096 in
theorem innerCandidates201_4_checked : candidates edge 4 1249 3391 innerTail201_5 best.bound = innerCandidates201_4 := by
  unfold candidates
  have row : edge 3391 = SearchData.row3391.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan201_4 : scan edge (search edge 3) 4 [1249] 1249 innerTail201_4 best = best := by
  rw [innerTail201_4, scan, if_neg (by decide), innerCandidates201_4_checked]
  change scan edge (search edge 3) 4 [1249] 1249 innerTail201_5
    (if 3 ≤ innerCandidates201_4.length then search edge 3 [1249, 3391] innerCandidates201_4 4640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan201_5

def innerTail201_3 : List Nat := 2551 :: innerTail201_4
def innerCandidates201_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates201_3_checked : candidates edge 4 1249 2551 innerTail201_4 best.bound = innerCandidates201_3 := by
  unfold candidates
  have row : edge 2551 = PrimePairCache.smallRow2551.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan201_3 : scan edge (search edge 3) 4 [1249] 1249 innerTail201_3 best = best := by
  rw [innerTail201_3, scan, if_neg (by decide), innerCandidates201_3_checked]
  change scan edge (search edge 3) 4 [1249] 1249 innerTail201_4
    (if 3 ≤ innerCandidates201_3.length then search edge 3 [1249, 2551] innerCandidates201_3 3800 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan201_4

def innerTail201_2 : List Nat := 2377 :: innerTail201_3
def innerCandidates201_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates201_2_checked : candidates edge 4 1249 2377 innerTail201_3 best.bound = innerCandidates201_2 := by
  unfold candidates
  have row : edge 2377 = PrimePairCache.smallRow2377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan201_2 : scan edge (search edge 3) 4 [1249] 1249 innerTail201_2 best = best := by
  rw [innerTail201_2, scan, if_neg (by decide), innerCandidates201_2_checked]
  change scan edge (search edge 3) 4 [1249] 1249 innerTail201_3
    (if 3 ≤ innerCandidates201_2.length then search edge 3 [1249, 2377] innerCandidates201_2 3626 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan201_3

def innerTail201_1 : List Nat := 2131 :: innerTail201_2
def innerCandidates201_1 : List Nat := [5563]
set_option maxRecDepth 4096 in
theorem innerCandidates201_1_checked : candidates edge 4 1249 2131 innerTail201_2 best.bound = innerCandidates201_1 := by
  unfold candidates
  have row : edge 2131 = PrimePairCache.smallRow2131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan201_1 : scan edge (search edge 3) 4 [1249] 1249 innerTail201_1 best = best := by
  rw [innerTail201_1, scan, if_neg (by decide), innerCandidates201_1_checked]
  change scan edge (search edge 3) 4 [1249] 1249 innerTail201_2
    (if 3 ≤ innerCandidates201_1.length then search edge 3 [1249, 2131] innerCandidates201_1 3380 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan201_2

def innerTail201_0 : List Nat := 1867 :: innerTail201_1
def innerCandidates201_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates201_0_checked : candidates edge 4 1249 1867 innerTail201_1 best.bound = innerCandidates201_0 := by
  unfold candidates
  have row : edge 1867 = PrimePairCache.smallRow1867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan201_0 : scan edge (search edge 3) 4 [1249] 1249 innerTail201_0 best = best := by
  rw [innerTail201_0, scan, if_neg (by decide), innerCandidates201_0_checked]
  change scan edge (search edge 3) 4 [1249] 1249 innerTail201_1
    (if 3 ≤ innerCandidates201_0.length then search edge 3 [1249, 1867] innerCandidates201_0 3116 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan201_1

theorem rootBranch201 : search edge 4 [1249] rootCandidates201 1249 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1249] 1249 innerTail201_0 best = best
  exact innerScan201_0

def rootCandidates200 : List Nat := [1543, 1549, 1657, 1741, 1753, 2341, 3019, 3187, 3253, 3307, 4027, 4993, 5557, 5743, 6271, 6571, 6673, 7789]
set_option maxRecDepth 4096 in
theorem rootCandidates200_checked : candidates edge 5 0 1237 tail201 best.bound = rootCandidates200 := by
  unfold candidates
  have row : edge 1237 = PrimePairCache.smallRow1237.contains := by funext q; rfl
  rw [row]
  decide

def innerTail200_18 : List Nat := []
theorem innerScan200_18 : scan edge (search edge 3) 4 [1237] 1237 innerTail200_18 best = best := rfl

def innerTail200_17 : List Nat := 7789 :: innerTail200_18
def innerCandidates200_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates200_17_checked : candidates edge 4 1237 7789 innerTail200_18 best.bound = innerCandidates200_17 := by
  unfold candidates
  have row : edge 7789 = SearchData.row7789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan200_17 : scan edge (search edge 3) 4 [1237] 1237 innerTail200_17 best = best := by
  rw [innerTail200_17, scan, if_neg (by decide), innerCandidates200_17_checked]
  change scan edge (search edge 3) 4 [1237] 1237 innerTail200_18
    (if 3 ≤ innerCandidates200_17.length then search edge 3 [1237, 7789] innerCandidates200_17 9026 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan200_18

def innerTail200_16 : List Nat := 6673 :: innerTail200_17
def innerCandidates200_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates200_16_checked : candidates edge 4 1237 6673 innerTail200_17 best.bound = innerCandidates200_16 := by
  unfold candidates
  have row : edge 6673 = SearchData.row6673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan200_16 : scan edge (search edge 3) 4 [1237] 1237 innerTail200_16 best = best := by
  rw [innerTail200_16, scan, if_neg (by decide), innerCandidates200_16_checked]
  change scan edge (search edge 3) 4 [1237] 1237 innerTail200_17
    (if 3 ≤ innerCandidates200_16.length then search edge 3 [1237, 6673] innerCandidates200_16 7910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan200_17

def innerTail200_15 : List Nat := 6571 :: innerTail200_16
def innerCandidates200_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates200_15_checked : candidates edge 4 1237 6571 innerTail200_16 best.bound = innerCandidates200_15 := by
  unfold candidates
  have row : edge 6571 = SearchData.row6571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan200_15 : scan edge (search edge 3) 4 [1237] 1237 innerTail200_15 best = best := by
  rw [innerTail200_15, scan, if_neg (by decide), innerCandidates200_15_checked]
  change scan edge (search edge 3) 4 [1237] 1237 innerTail200_16
    (if 3 ≤ innerCandidates200_15.length then search edge 3 [1237, 6571] innerCandidates200_15 7808 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan200_16

def innerTail200_14 : List Nat := 6271 :: innerTail200_15
def innerCandidates200_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates200_14_checked : candidates edge 4 1237 6271 innerTail200_15 best.bound = innerCandidates200_14 := by
  unfold candidates
  have row : edge 6271 = SearchData.row6271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan200_14 : scan edge (search edge 3) 4 [1237] 1237 innerTail200_14 best = best := by
  rw [innerTail200_14, scan, if_neg (by decide), innerCandidates200_14_checked]
  change scan edge (search edge 3) 4 [1237] 1237 innerTail200_15
    (if 3 ≤ innerCandidates200_14.length then search edge 3 [1237, 6271] innerCandidates200_14 7508 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan200_15

def innerTail200_13 : List Nat := 5743 :: innerTail200_14
def innerCandidates200_13 : List Nat := [6271]
set_option maxRecDepth 4096 in
theorem innerCandidates200_13_checked : candidates edge 4 1237 5743 innerTail200_14 best.bound = innerCandidates200_13 := by
  unfold candidates
  have row : edge 5743 = SearchData.row5743.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan200_13 : scan edge (search edge 3) 4 [1237] 1237 innerTail200_13 best = best := by
  rw [innerTail200_13, scan, if_neg (by decide), innerCandidates200_13_checked]
  change scan edge (search edge 3) 4 [1237] 1237 innerTail200_14
    (if 3 ≤ innerCandidates200_13.length then search edge 3 [1237, 5743] innerCandidates200_13 6980 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan200_14

def innerTail200_12 : List Nat := 5557 :: innerTail200_13
def innerCandidates200_12 : List Nat := [7789]
set_option maxRecDepth 4096 in
theorem innerCandidates200_12_checked : candidates edge 4 1237 5557 innerTail200_13 best.bound = innerCandidates200_12 := by
  unfold candidates
  have row : edge 5557 = SearchData.row5557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan200_12 : scan edge (search edge 3) 4 [1237] 1237 innerTail200_12 best = best := by
  rw [innerTail200_12, scan, if_neg (by decide), innerCandidates200_12_checked]
  change scan edge (search edge 3) 4 [1237] 1237 innerTail200_13
    (if 3 ≤ innerCandidates200_12.length then search edge 3 [1237, 5557] innerCandidates200_12 6794 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan200_13

def innerTail200_11 : List Nat := 4993 :: innerTail200_12
def innerCandidates200_11 : List Nat := [6571]
set_option maxRecDepth 4096 in
theorem innerCandidates200_11_checked : candidates edge 4 1237 4993 innerTail200_12 best.bound = innerCandidates200_11 := by
  unfold candidates
  have row : edge 4993 = SearchData.row4993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan200_11 : scan edge (search edge 3) 4 [1237] 1237 innerTail200_11 best = best := by
  rw [innerTail200_11, scan, if_neg (by decide), innerCandidates200_11_checked]
  change scan edge (search edge 3) 4 [1237] 1237 innerTail200_12
    (if 3 ≤ innerCandidates200_11.length then search edge 3 [1237, 4993] innerCandidates200_11 6230 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan200_12

def innerTail200_10 : List Nat := 4027 :: innerTail200_11
def innerCandidates200_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates200_10_checked : candidates edge 4 1237 4027 innerTail200_11 best.bound = innerCandidates200_10 := by
  unfold candidates
  have row : edge 4027 = SearchData.row4027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan200_10 : scan edge (search edge 3) 4 [1237] 1237 innerTail200_10 best = best := by
  rw [innerTail200_10, scan, if_neg (by decide), innerCandidates200_10_checked]
  change scan edge (search edge 3) 4 [1237] 1237 innerTail200_11
    (if 3 ≤ innerCandidates200_10.length then search edge 3 [1237, 4027] innerCandidates200_10 5264 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan200_11

def innerTail200_9 : List Nat := 3307 :: innerTail200_10
def innerCandidates200_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates200_9_checked : candidates edge 4 1237 3307 innerTail200_10 best.bound = innerCandidates200_9 := by
  unfold candidates
  have row : edge 3307 = SearchData.row3307.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan200_9 : scan edge (search edge 3) 4 [1237] 1237 innerTail200_9 best = best := by
  rw [innerTail200_9, scan, if_neg (by decide), innerCandidates200_9_checked]
  change scan edge (search edge 3) 4 [1237] 1237 innerTail200_10
    (if 3 ≤ innerCandidates200_9.length then search edge 3 [1237, 3307] innerCandidates200_9 4544 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan200_10

def innerTail200_8 : List Nat := 3253 :: innerTail200_9
def innerCandidates200_8 : List Nat := [5557, 6571]
set_option maxRecDepth 4096 in
theorem innerCandidates200_8_checked : candidates edge 4 1237 3253 innerTail200_9 best.bound = innerCandidates200_8 := by
  unfold candidates
  have row : edge 3253 = SearchData.row3253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan200_8 : scan edge (search edge 3) 4 [1237] 1237 innerTail200_8 best = best := by
  rw [innerTail200_8, scan, if_neg (by decide), innerCandidates200_8_checked]
  change scan edge (search edge 3) 4 [1237] 1237 innerTail200_9
    (if 3 ≤ innerCandidates200_8.length then search edge 3 [1237, 3253] innerCandidates200_8 4490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan200_9

def innerTail200_7 : List Nat := 3187 :: innerTail200_8
def innerCandidates200_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates200_7_checked : candidates edge 4 1237 3187 innerTail200_8 best.bound = innerCandidates200_7 := by
  unfold candidates
  have row : edge 3187 = SearchData.row3187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan200_7 : scan edge (search edge 3) 4 [1237] 1237 innerTail200_7 best = best := by
  rw [innerTail200_7, scan, if_neg (by decide), innerCandidates200_7_checked]
  change scan edge (search edge 3) 4 [1237] 1237 innerTail200_8
    (if 3 ≤ innerCandidates200_7.length then search edge 3 [1237, 3187] innerCandidates200_7 4424 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan200_8

def innerTail200_6 : List Nat := 3019 :: innerTail200_7
def innerCandidates200_6 : List Nat := [6571]
set_option maxRecDepth 4096 in
theorem innerCandidates200_6_checked : candidates edge 4 1237 3019 innerTail200_7 best.bound = innerCandidates200_6 := by
  unfold candidates
  have row : edge 3019 = SearchData.row3019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan200_6 : scan edge (search edge 3) 4 [1237] 1237 innerTail200_6 best = best := by
  rw [innerTail200_6, scan, if_neg (by decide), innerCandidates200_6_checked]
  change scan edge (search edge 3) 4 [1237] 1237 innerTail200_7
    (if 3 ≤ innerCandidates200_6.length then search edge 3 [1237, 3019] innerCandidates200_6 4256 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan200_7

def innerTail200_5 : List Nat := 2341 :: innerTail200_6
def innerCandidates200_5 : List Nat := [6571]
set_option maxRecDepth 4096 in
theorem innerCandidates200_5_checked : candidates edge 4 1237 2341 innerTail200_6 best.bound = innerCandidates200_5 := by
  unfold candidates
  have row : edge 2341 = PrimePairCache.smallRow2341.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan200_5 : scan edge (search edge 3) 4 [1237] 1237 innerTail200_5 best = best := by
  rw [innerTail200_5, scan, if_neg (by decide), innerCandidates200_5_checked]
  change scan edge (search edge 3) 4 [1237] 1237 innerTail200_6
    (if 3 ≤ innerCandidates200_5.length then search edge 3 [1237, 2341] innerCandidates200_5 3578 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan200_6

def innerTail200_4 : List Nat := 1753 :: innerTail200_5
def innerCandidates200_4 : List Nat := [7789]
set_option maxRecDepth 4096 in
theorem innerCandidates200_4_checked : candidates edge 4 1237 1753 innerTail200_5 best.bound = innerCandidates200_4 := by
  unfold candidates
  have row : edge 1753 = PrimePairCache.smallRow1753.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan200_4 : scan edge (search edge 3) 4 [1237] 1237 innerTail200_4 best = best := by
  rw [innerTail200_4, scan, if_neg (by decide), innerCandidates200_4_checked]
  change scan edge (search edge 3) 4 [1237] 1237 innerTail200_5
    (if 3 ≤ innerCandidates200_4.length then search edge 3 [1237, 1753] innerCandidates200_4 2990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan200_5

def innerTail200_3 : List Nat := 1741 :: innerTail200_4
def innerCandidates200_3 : List Nat := [1753, 3307, 5557]
set_option maxRecDepth 4096 in
theorem innerCandidates200_3_checked : candidates edge 4 1237 1741 innerTail200_4 best.bound = innerCandidates200_3 := by
  unfold candidates
  have row : edge 1741 = PrimePairCache.smallRow1741.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild200_3 : search edge 3 [1237, 1741] innerCandidates200_3 2978 best = best := by decide

theorem innerScan200_3 : scan edge (search edge 3) 4 [1237] 1237 innerTail200_3 best = best := by
  rw [innerTail200_3, scan, if_neg (by decide), innerCandidates200_3_checked]
  change scan edge (search edge 3) 4 [1237] 1237 innerTail200_4
    (if 3 ≤ innerCandidates200_3.length then search edge 3 [1237, 1741] innerCandidates200_3 2978 best else best) = best
  rw [if_pos (by decide), innerChild200_3]
  exact innerScan200_4

def innerTail200_2 : List Nat := 1657 :: innerTail200_3
def innerCandidates200_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates200_2_checked : candidates edge 4 1237 1657 innerTail200_3 best.bound = innerCandidates200_2 := by
  unfold candidates
  have row : edge 1657 = PrimePairCache.smallRow1657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan200_2 : scan edge (search edge 3) 4 [1237] 1237 innerTail200_2 best = best := by
  rw [innerTail200_2, scan, if_neg (by decide), innerCandidates200_2_checked]
  change scan edge (search edge 3) 4 [1237] 1237 innerTail200_3
    (if 3 ≤ innerCandidates200_2.length then search edge 3 [1237, 1657] innerCandidates200_2 2894 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan200_3

def innerTail200_1 : List Nat := 1549 :: innerTail200_2
def innerCandidates200_1 : List Nat := [3019, 3253]
set_option maxRecDepth 4096 in
theorem innerCandidates200_1_checked : candidates edge 4 1237 1549 innerTail200_2 best.bound = innerCandidates200_1 := by
  unfold candidates
  have row : edge 1549 = PrimePairCache.smallRow1549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan200_1 : scan edge (search edge 3) 4 [1237] 1237 innerTail200_1 best = best := by
  rw [innerTail200_1, scan, if_neg (by decide), innerCandidates200_1_checked]
  change scan edge (search edge 3) 4 [1237] 1237 innerTail200_2
    (if 3 ≤ innerCandidates200_1.length then search edge 3 [1237, 1549] innerCandidates200_1 2786 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan200_2

def innerTail200_0 : List Nat := 1543 :: innerTail200_1
def innerCandidates200_0 : List Nat := [6571]
set_option maxRecDepth 4096 in
theorem innerCandidates200_0_checked : candidates edge 4 1237 1543 innerTail200_1 best.bound = innerCandidates200_0 := by
  unfold candidates
  have row : edge 1543 = PrimePairCache.smallRow1543.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan200_0 : scan edge (search edge 3) 4 [1237] 1237 innerTail200_0 best = best := by
  rw [innerTail200_0, scan, if_neg (by decide), innerCandidates200_0_checked]
  change scan edge (search edge 3) 4 [1237] 1237 innerTail200_1
    (if 3 ≤ innerCandidates200_0.length then search edge 3 [1237, 1543] innerCandidates200_0 2780 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan200_1

theorem rootBranch200 : search edge 4 [1237] rootCandidates200 1237 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1237] 1237 innerTail200_0 best = best
  exact innerScan200_0

def rootCandidates199 : List Nat := [1777, 1951, 2143, 2689, 2917, 4231, 4243, 4327, 4483, 4507, 4597, 4789, 5323, 5449, 5647, 5659, 5881, 6037, 6091, 6229, 6529, 6949, 7297, 7309, 8221]
set_option maxRecDepth 4096 in
theorem rootCandidates199_checked : candidates edge 5 0 1231 tail200 best.bound = rootCandidates199 := by
  unfold candidates
  have row : edge 1231 = PrimePairCache.smallRow1231.contains := by funext q; rfl
  rw [row]
  decide

def innerTail199_25 : List Nat := []
theorem innerScan199_25 : scan edge (search edge 3) 4 [1231] 1231 innerTail199_25 best = best := rfl

def innerTail199_24 : List Nat := 8221 :: innerTail199_25
def innerCandidates199_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates199_24_checked : candidates edge 4 1231 8221 innerTail199_25 best.bound = innerCandidates199_24 := by
  unfold candidates
  have row : edge 8221 = SearchData.row8221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan199_24 : scan edge (search edge 3) 4 [1231] 1231 innerTail199_24 best = best := by
  rw [innerTail199_24, scan, if_neg (by decide), innerCandidates199_24_checked]
  change scan edge (search edge 3) 4 [1231] 1231 innerTail199_25
    (if 3 ≤ innerCandidates199_24.length then search edge 3 [1231, 8221] innerCandidates199_24 9452 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan199_25

def innerTail199_23 : List Nat := 7309 :: innerTail199_24
def innerCandidates199_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates199_23_checked : candidates edge 4 1231 7309 innerTail199_24 best.bound = innerCandidates199_23 := by
  unfold candidates
  have row : edge 7309 = SearchData.row7309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan199_23 : scan edge (search edge 3) 4 [1231] 1231 innerTail199_23 best = best := by
  rw [innerTail199_23, scan, if_neg (by decide), innerCandidates199_23_checked]
  change scan edge (search edge 3) 4 [1231] 1231 innerTail199_24
    (if 3 ≤ innerCandidates199_23.length then search edge 3 [1231, 7309] innerCandidates199_23 8540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan199_24

def innerTail199_22 : List Nat := 7297 :: innerTail199_23
def innerCandidates199_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates199_22_checked : candidates edge 4 1231 7297 innerTail199_23 best.bound = innerCandidates199_22 := by
  unfold candidates
  have row : edge 7297 = SearchData.row7297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan199_22 : scan edge (search edge 3) 4 [1231] 1231 innerTail199_22 best = best := by
  rw [innerTail199_22, scan, if_neg (by decide), innerCandidates199_22_checked]
  change scan edge (search edge 3) 4 [1231] 1231 innerTail199_23
    (if 3 ≤ innerCandidates199_22.length then search edge 3 [1231, 7297] innerCandidates199_22 8528 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan199_23

def innerTail199_21 : List Nat := 6949 :: innerTail199_22
def innerCandidates199_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates199_21_checked : candidates edge 4 1231 6949 innerTail199_22 best.bound = innerCandidates199_21 := by
  unfold candidates
  have row : edge 6949 = SearchData.row6949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan199_21 : scan edge (search edge 3) 4 [1231] 1231 innerTail199_21 best = best := by
  rw [innerTail199_21, scan, if_neg (by decide), innerCandidates199_21_checked]
  change scan edge (search edge 3) 4 [1231] 1231 innerTail199_22
    (if 3 ≤ innerCandidates199_21.length then search edge 3 [1231, 6949] innerCandidates199_21 8180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan199_22

def innerTail199_20 : List Nat := 6529 :: innerTail199_21
def innerCandidates199_20 : List Nat := [8221]
set_option maxRecDepth 4096 in
theorem innerCandidates199_20_checked : candidates edge 4 1231 6529 innerTail199_21 best.bound = innerCandidates199_20 := by
  unfold candidates
  have row : edge 6529 = SearchData.row6529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan199_20 : scan edge (search edge 3) 4 [1231] 1231 innerTail199_20 best = best := by
  rw [innerTail199_20, scan, if_neg (by decide), innerCandidates199_20_checked]
  change scan edge (search edge 3) 4 [1231] 1231 innerTail199_21
    (if 3 ≤ innerCandidates199_20.length then search edge 3 [1231, 6529] innerCandidates199_20 7760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan199_21

def innerTail199_19 : List Nat := 6229 :: innerTail199_20
def innerCandidates199_19 : List Nat := [7297]
set_option maxRecDepth 4096 in
theorem innerCandidates199_19_checked : candidates edge 4 1231 6229 innerTail199_20 best.bound = innerCandidates199_19 := by
  unfold candidates
  have row : edge 6229 = SearchData.row6229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan199_19 : scan edge (search edge 3) 4 [1231] 1231 innerTail199_19 best = best := by
  rw [innerTail199_19, scan, if_neg (by decide), innerCandidates199_19_checked]
  change scan edge (search edge 3) 4 [1231] 1231 innerTail199_20
    (if 3 ≤ innerCandidates199_19.length then search edge 3 [1231, 6229] innerCandidates199_19 7460 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan199_20

def innerTail199_18 : List Nat := 6091 :: innerTail199_19
def innerCandidates199_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates199_18_checked : candidates edge 4 1231 6091 innerTail199_19 best.bound = innerCandidates199_18 := by
  unfold candidates
  have row : edge 6091 = SearchData.row6091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan199_18 : scan edge (search edge 3) 4 [1231] 1231 innerTail199_18 best = best := by
  rw [innerTail199_18, scan, if_neg (by decide), innerCandidates199_18_checked]
  change scan edge (search edge 3) 4 [1231] 1231 innerTail199_19
    (if 3 ≤ innerCandidates199_18.length then search edge 3 [1231, 6091] innerCandidates199_18 7322 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan199_19

def innerTail199_17 : List Nat := 6037 :: innerTail199_18
def innerCandidates199_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates199_17_checked : candidates edge 4 1231 6037 innerTail199_18 best.bound = innerCandidates199_17 := by
  unfold candidates
  have row : edge 6037 = SearchData.row6037.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan199_17 : scan edge (search edge 3) 4 [1231] 1231 innerTail199_17 best = best := by
  rw [innerTail199_17, scan, if_neg (by decide), innerCandidates199_17_checked]
  change scan edge (search edge 3) 4 [1231] 1231 innerTail199_18
    (if 3 ≤ innerCandidates199_17.length then search edge 3 [1231, 6037] innerCandidates199_17 7268 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan199_18

def innerTail199_16 : List Nat := 5881 :: innerTail199_17
def innerCandidates199_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates199_16_checked : candidates edge 4 1231 5881 innerTail199_17 best.bound = innerCandidates199_16 := by
  unfold candidates
  have row : edge 5881 = SearchData.row5881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan199_16 : scan edge (search edge 3) 4 [1231] 1231 innerTail199_16 best = best := by
  rw [innerTail199_16, scan, if_neg (by decide), innerCandidates199_16_checked]
  change scan edge (search edge 3) 4 [1231] 1231 innerTail199_17
    (if 3 ≤ innerCandidates199_16.length then search edge 3 [1231, 5881] innerCandidates199_16 7112 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan199_17

def innerTail199_15 : List Nat := 5659 :: innerTail199_16
def innerCandidates199_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates199_15_checked : candidates edge 4 1231 5659 innerTail199_16 best.bound = innerCandidates199_15 := by
  unfold candidates
  have row : edge 5659 = SearchData.row5659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan199_15 : scan edge (search edge 3) 4 [1231] 1231 innerTail199_15 best = best := by
  rw [innerTail199_15, scan, if_neg (by decide), innerCandidates199_15_checked]
  change scan edge (search edge 3) 4 [1231] 1231 innerTail199_16
    (if 3 ≤ innerCandidates199_15.length then search edge 3 [1231, 5659] innerCandidates199_15 6890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan199_16

def innerTail199_14 : List Nat := 5647 :: innerTail199_15
def innerCandidates199_14 : List Nat := [5659]
set_option maxRecDepth 4096 in
theorem innerCandidates199_14_checked : candidates edge 4 1231 5647 innerTail199_15 best.bound = innerCandidates199_14 := by
  unfold candidates
  have row : edge 5647 = SearchData.row5647.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan199_14 : scan edge (search edge 3) 4 [1231] 1231 innerTail199_14 best = best := by
  rw [innerTail199_14, scan, if_neg (by decide), innerCandidates199_14_checked]
  change scan edge (search edge 3) 4 [1231] 1231 innerTail199_15
    (if 3 ≤ innerCandidates199_14.length then search edge 3 [1231, 5647] innerCandidates199_14 6878 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan199_15

def innerTail199_13 : List Nat := 5449 :: innerTail199_14
def innerCandidates199_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates199_13_checked : candidates edge 4 1231 5449 innerTail199_14 best.bound = innerCandidates199_13 := by
  unfold candidates
  have row : edge 5449 = SearchData.row5449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan199_13 : scan edge (search edge 3) 4 [1231] 1231 innerTail199_13 best = best := by
  rw [innerTail199_13, scan, if_neg (by decide), innerCandidates199_13_checked]
  change scan edge (search edge 3) 4 [1231] 1231 innerTail199_14
    (if 3 ≤ innerCandidates199_13.length then search edge 3 [1231, 5449] innerCandidates199_13 6680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan199_14

def innerTail199_12 : List Nat := 5323 :: innerTail199_13
def innerCandidates199_12 : List Nat := [5659, 6037]
set_option maxRecDepth 4096 in
theorem innerCandidates199_12_checked : candidates edge 4 1231 5323 innerTail199_13 best.bound = innerCandidates199_12 := by
  unfold candidates
  have row : edge 5323 = SearchData.row5323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan199_12 : scan edge (search edge 3) 4 [1231] 1231 innerTail199_12 best = best := by
  rw [innerTail199_12, scan, if_neg (by decide), innerCandidates199_12_checked]
  change scan edge (search edge 3) 4 [1231] 1231 innerTail199_13
    (if 3 ≤ innerCandidates199_12.length then search edge 3 [1231, 5323] innerCandidates199_12 6554 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan199_13

def innerTail199_11 : List Nat := 4789 :: innerTail199_12
def innerCandidates199_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates199_11_checked : candidates edge 4 1231 4789 innerTail199_12 best.bound = innerCandidates199_11 := by
  unfold candidates
  have row : edge 4789 = SearchData.row4789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan199_11 : scan edge (search edge 3) 4 [1231] 1231 innerTail199_11 best = best := by
  rw [innerTail199_11, scan, if_neg (by decide), innerCandidates199_11_checked]
  change scan edge (search edge 3) 4 [1231] 1231 innerTail199_12
    (if 3 ≤ innerCandidates199_11.length then search edge 3 [1231, 4789] innerCandidates199_11 6020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan199_12

def innerTail199_10 : List Nat := 4597 :: innerTail199_11
def innerCandidates199_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates199_10_checked : candidates edge 4 1231 4597 innerTail199_11 best.bound = innerCandidates199_10 := by
  unfold candidates
  have row : edge 4597 = SearchData.row4597.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan199_10 : scan edge (search edge 3) 4 [1231] 1231 innerTail199_10 best = best := by
  rw [innerTail199_10, scan, if_neg (by decide), innerCandidates199_10_checked]
  change scan edge (search edge 3) 4 [1231] 1231 innerTail199_11
    (if 3 ≤ innerCandidates199_10.length then search edge 3 [1231, 4597] innerCandidates199_10 5828 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan199_11

def innerTail199_9 : List Nat := 4507 :: innerTail199_10
def innerCandidates199_9 : List Nat := [4789, 5881, 6091]
set_option maxRecDepth 4096 in
theorem innerCandidates199_9_checked : candidates edge 4 1231 4507 innerTail199_10 best.bound = innerCandidates199_9 := by
  unfold candidates
  have row : edge 4507 = SearchData.row4507.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild199_9 : search edge 3 [1231, 4507] innerCandidates199_9 5738 best = best := by decide

theorem innerScan199_9 : scan edge (search edge 3) 4 [1231] 1231 innerTail199_9 best = best := by
  rw [innerTail199_9, scan, if_neg (by decide), innerCandidates199_9_checked]
  change scan edge (search edge 3) 4 [1231] 1231 innerTail199_10
    (if 3 ≤ innerCandidates199_9.length then search edge 3 [1231, 4507] innerCandidates199_9 5738 best else best) = best
  rw [if_pos (by decide), innerChild199_9]
  exact innerScan199_10

def innerTail199_8 : List Nat := 4483 :: innerTail199_9
def innerCandidates199_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates199_8_checked : candidates edge 4 1231 4483 innerTail199_9 best.bound = innerCandidates199_8 := by
  unfold candidates
  have row : edge 4483 = SearchData.row4483.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan199_8 : scan edge (search edge 3) 4 [1231] 1231 innerTail199_8 best = best := by
  rw [innerTail199_8, scan, if_neg (by decide), innerCandidates199_8_checked]
  change scan edge (search edge 3) 4 [1231] 1231 innerTail199_9
    (if 3 ≤ innerCandidates199_8.length then search edge 3 [1231, 4483] innerCandidates199_8 5714 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan199_9

def innerTail199_7 : List Nat := 4327 :: innerTail199_8
def innerCandidates199_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates199_7_checked : candidates edge 4 1231 4327 innerTail199_8 best.bound = innerCandidates199_7 := by
  unfold candidates
  have row : edge 4327 = SearchData.row4327.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan199_7 : scan edge (search edge 3) 4 [1231] 1231 innerTail199_7 best = best := by
  rw [innerTail199_7, scan, if_neg (by decide), innerCandidates199_7_checked]
  change scan edge (search edge 3) 4 [1231] 1231 innerTail199_8
    (if 3 ≤ innerCandidates199_7.length then search edge 3 [1231, 4327] innerCandidates199_7 5558 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan199_8

def innerTail199_6 : List Nat := 4243 :: innerTail199_7
def innerCandidates199_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates199_6_checked : candidates edge 4 1231 4243 innerTail199_7 best.bound = innerCandidates199_6 := by
  unfold candidates
  have row : edge 4243 = SearchData.row4243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan199_6 : scan edge (search edge 3) 4 [1231] 1231 innerTail199_6 best = best := by
  rw [innerTail199_6, scan, if_neg (by decide), innerCandidates199_6_checked]
  change scan edge (search edge 3) 4 [1231] 1231 innerTail199_7
    (if 3 ≤ innerCandidates199_6.length then search edge 3 [1231, 4243] innerCandidates199_6 5474 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan199_7

def innerTail199_5 : List Nat := 4231 :: innerTail199_6
def innerCandidates199_5 : List Nat := [6037, 6529]
set_option maxRecDepth 4096 in
theorem innerCandidates199_5_checked : candidates edge 4 1231 4231 innerTail199_6 best.bound = innerCandidates199_5 := by
  unfold candidates
  have row : edge 4231 = SearchData.row4231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan199_5 : scan edge (search edge 3) 4 [1231] 1231 innerTail199_5 best = best := by
  rw [innerTail199_5, scan, if_neg (by decide), innerCandidates199_5_checked]
  change scan edge (search edge 3) 4 [1231] 1231 innerTail199_6
    (if 3 ≤ innerCandidates199_5.length then search edge 3 [1231, 4231] innerCandidates199_5 5462 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan199_6

def innerTail199_4 : List Nat := 2917 :: innerTail199_5
def innerCandidates199_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates199_4_checked : candidates edge 4 1231 2917 innerTail199_5 best.bound = innerCandidates199_4 := by
  unfold candidates
  have row : edge 2917 = SearchData.row2917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan199_4 : scan edge (search edge 3) 4 [1231] 1231 innerTail199_4 best = best := by
  rw [innerTail199_4, scan, if_neg (by decide), innerCandidates199_4_checked]
  change scan edge (search edge 3) 4 [1231] 1231 innerTail199_5
    (if 3 ≤ innerCandidates199_4.length then search edge 3 [1231, 2917] innerCandidates199_4 4148 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan199_5

def innerTail199_3 : List Nat := 2689 :: innerTail199_4
def innerCandidates199_3 : List Nat := [4507]
set_option maxRecDepth 4096 in
theorem innerCandidates199_3_checked : candidates edge 4 1231 2689 innerTail199_4 best.bound = innerCandidates199_3 := by
  unfold candidates
  have row : edge 2689 = SearchData.row2689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan199_3 : scan edge (search edge 3) 4 [1231] 1231 innerTail199_3 best = best := by
  rw [innerTail199_3, scan, if_neg (by decide), innerCandidates199_3_checked]
  change scan edge (search edge 3) 4 [1231] 1231 innerTail199_4
    (if 3 ≤ innerCandidates199_3.length then search edge 3 [1231, 2689] innerCandidates199_3 3920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan199_4

def innerTail199_2 : List Nat := 2143 :: innerTail199_3
def innerCandidates199_2 : List Nat := [6229]
set_option maxRecDepth 4096 in
theorem innerCandidates199_2_checked : candidates edge 4 1231 2143 innerTail199_3 best.bound = innerCandidates199_2 := by
  unfold candidates
  have row : edge 2143 = PrimePairCache.smallRow2143.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan199_2 : scan edge (search edge 3) 4 [1231] 1231 innerTail199_2 best = best := by
  rw [innerTail199_2, scan, if_neg (by decide), innerCandidates199_2_checked]
  change scan edge (search edge 3) 4 [1231] 1231 innerTail199_3
    (if 3 ≤ innerCandidates199_2.length then search edge 3 [1231, 2143] innerCandidates199_2 3374 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan199_3

def innerTail199_1 : List Nat := 1951 :: innerTail199_2
def innerCandidates199_1 : List Nat := [6037]
set_option maxRecDepth 4096 in
theorem innerCandidates199_1_checked : candidates edge 4 1231 1951 innerTail199_2 best.bound = innerCandidates199_1 := by
  unfold candidates
  have row : edge 1951 = PrimePairCache.smallRow1951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan199_1 : scan edge (search edge 3) 4 [1231] 1231 innerTail199_1 best = best := by
  rw [innerTail199_1, scan, if_neg (by decide), innerCandidates199_1_checked]
  change scan edge (search edge 3) 4 [1231] 1231 innerTail199_2
    (if 3 ≤ innerCandidates199_1.length then search edge 3 [1231, 1951] innerCandidates199_1 3182 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan199_2

def innerTail199_0 : List Nat := 1777 :: innerTail199_1
def innerCandidates199_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates199_0_checked : candidates edge 4 1231 1777 innerTail199_1 best.bound = innerCandidates199_0 := by
  unfold candidates
  have row : edge 1777 = PrimePairCache.smallRow1777.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan199_0 : scan edge (search edge 3) 4 [1231] 1231 innerTail199_0 best = best := by
  rw [innerTail199_0, scan, if_neg (by decide), innerCandidates199_0_checked]
  change scan edge (search edge 3) 4 [1231] 1231 innerTail199_1
    (if 3 ≤ innerCandidates199_0.length then search edge 3 [1231, 1777] innerCandidates199_0 3008 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan199_1

theorem rootBranch199 : search edge 4 [1231] rootCandidates199 1231 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1231] 1231 innerTail199_0 best = best
  exact innerScan199_0

end ProjectEulerSolutions.PrimePairUpper.Search
