import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates562 : List Nat := [4783, 4909, 4999, 5953, 6793]
set_option maxRecDepth 4096 in
theorem rootCandidates562_checked : candidates edge 5 0 4099 tail563 best.bound = rootCandidates562 := by
  unfold candidates
  have row : edge 4099 = SearchData.row4099.contains := by funext q; rfl
  rw [row]
  decide

def innerTail562_5 : List Nat := []
theorem innerScan562_5 : scan edge (search edge 3) 4 [4099] 4099 innerTail562_5 best = best := rfl

def innerTail562_4 : List Nat := 6793 :: innerTail562_5
def innerCandidates562_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates562_4_checked : candidates edge 4 4099 6793 innerTail562_5 best.bound = innerCandidates562_4 := by
  unfold candidates
  have row : edge 6793 = SearchData.row6793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan562_4 : scan edge (search edge 3) 4 [4099] 4099 innerTail562_4 best = best := by
  rw [innerTail562_4, scan, if_neg (by decide), innerCandidates562_4_checked]
  change scan edge (search edge 3) 4 [4099] 4099 innerTail562_5
    (if 3 ≤ innerCandidates562_4.length then search edge 3 [4099, 6793] innerCandidates562_4 10892 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan562_5

def innerTail562_3 : List Nat := 5953 :: innerTail562_4
def innerCandidates562_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates562_3_checked : candidates edge 4 4099 5953 innerTail562_4 best.bound = innerCandidates562_3 := by
  unfold candidates
  have row : edge 5953 = SearchData.row5953.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan562_3 : scan edge (search edge 3) 4 [4099] 4099 innerTail562_3 best = best := by
  rw [innerTail562_3, scan, if_neg (by decide), innerCandidates562_3_checked]
  change scan edge (search edge 3) 4 [4099] 4099 innerTail562_4
    (if 3 ≤ innerCandidates562_3.length then search edge 3 [4099, 5953] innerCandidates562_3 10052 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan562_4

def innerTail562_2 : List Nat := 4999 :: innerTail562_3
def innerCandidates562_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates562_2_checked : candidates edge 4 4099 4999 innerTail562_3 best.bound = innerCandidates562_2 := by
  unfold candidates
  have row : edge 4999 = SearchData.row4999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan562_2 : scan edge (search edge 3) 4 [4099] 4099 innerTail562_2 best = best := by
  rw [innerTail562_2, scan, if_neg (by decide), innerCandidates562_2_checked]
  change scan edge (search edge 3) 4 [4099] 4099 innerTail562_3
    (if 3 ≤ innerCandidates562_2.length then search edge 3 [4099, 4999] innerCandidates562_2 9098 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan562_3

def innerTail562_1 : List Nat := 4909 :: innerTail562_2
def innerCandidates562_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates562_1_checked : candidates edge 4 4099 4909 innerTail562_2 best.bound = innerCandidates562_1 := by
  unfold candidates
  have row : edge 4909 = SearchData.row4909.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan562_1 : scan edge (search edge 3) 4 [4099] 4099 innerTail562_1 best = best := by
  rw [innerTail562_1, scan, if_neg (by decide), innerCandidates562_1_checked]
  change scan edge (search edge 3) 4 [4099] 4099 innerTail562_2
    (if 3 ≤ innerCandidates562_1.length then search edge 3 [4099, 4909] innerCandidates562_1 9008 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan562_2

def innerTail562_0 : List Nat := 4783 :: innerTail562_1
def innerCandidates562_0 : List Nat := [4999]
set_option maxRecDepth 4096 in
theorem innerCandidates562_0_checked : candidates edge 4 4099 4783 innerTail562_1 best.bound = innerCandidates562_0 := by
  unfold candidates
  have row : edge 4783 = SearchData.row4783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan562_0 : scan edge (search edge 3) 4 [4099] 4099 innerTail562_0 best = best := by
  rw [innerTail562_0, scan, if_neg (by decide), innerCandidates562_0_checked]
  change scan edge (search edge 3) 4 [4099] 4099 innerTail562_1
    (if 3 ≤ innerCandidates562_0.length then search edge 3 [4099, 4783] innerCandidates562_0 8882 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan562_1

theorem rootBranch562 : search edge 4 [4099] rootCandidates562 4099 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4099] 4099 innerTail562_0 best = best
  exact innerScan562_0

def rootCandidates561 : List Nat := [4567, 5077, 5119, 5659, 5749, 6823, 6991, 7321, 7933, 8167]
set_option maxRecDepth 4096 in
theorem rootCandidates561_checked : candidates edge 5 0 4093 tail562 best.bound = rootCandidates561 := by
  unfold candidates
  have row : edge 4093 = SearchData.row4093.contains := by funext q; rfl
  rw [row]
  decide

def innerTail561_10 : List Nat := []
theorem innerScan561_10 : scan edge (search edge 3) 4 [4093] 4093 innerTail561_10 best = best := rfl

def innerTail561_9 : List Nat := 8167 :: innerTail561_10
def innerCandidates561_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates561_9_checked : candidates edge 4 4093 8167 innerTail561_10 best.bound = innerCandidates561_9 := by
  unfold candidates
  have row : edge 8167 = SearchData.row8167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan561_9 : scan edge (search edge 3) 4 [4093] 4093 innerTail561_9 best = best := by
  rw [innerTail561_9, scan, if_neg (by decide), innerCandidates561_9_checked]
  change scan edge (search edge 3) 4 [4093] 4093 innerTail561_10
    (if 3 ≤ innerCandidates561_9.length then search edge 3 [4093, 8167] innerCandidates561_9 12260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan561_10

def innerTail561_8 : List Nat := 7933 :: innerTail561_9
def innerCandidates561_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates561_8_checked : candidates edge 4 4093 7933 innerTail561_9 best.bound = innerCandidates561_8 := by
  unfold candidates
  have row : edge 7933 = SearchData.row7933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan561_8 : scan edge (search edge 3) 4 [4093] 4093 innerTail561_8 best = best := by
  rw [innerTail561_8, scan, if_neg (by decide), innerCandidates561_8_checked]
  change scan edge (search edge 3) 4 [4093] 4093 innerTail561_9
    (if 3 ≤ innerCandidates561_8.length then search edge 3 [4093, 7933] innerCandidates561_8 12026 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan561_9

def innerTail561_7 : List Nat := 7321 :: innerTail561_8
def innerCandidates561_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates561_7_checked : candidates edge 4 4093 7321 innerTail561_8 best.bound = innerCandidates561_7 := by
  unfold candidates
  have row : edge 7321 = SearchData.row7321.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan561_7 : scan edge (search edge 3) 4 [4093] 4093 innerTail561_7 best = best := by
  rw [innerTail561_7, scan, if_neg (by decide), innerCandidates561_7_checked]
  change scan edge (search edge 3) 4 [4093] 4093 innerTail561_8
    (if 3 ≤ innerCandidates561_7.length then search edge 3 [4093, 7321] innerCandidates561_7 11414 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan561_8

def innerTail561_6 : List Nat := 6991 :: innerTail561_7
def innerCandidates561_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates561_6_checked : candidates edge 4 4093 6991 innerTail561_7 best.bound = innerCandidates561_6 := by
  unfold candidates
  have row : edge 6991 = SearchData.row6991.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan561_6 : scan edge (search edge 3) 4 [4093] 4093 innerTail561_6 best = best := by
  rw [innerTail561_6, scan, if_neg (by decide), innerCandidates561_6_checked]
  change scan edge (search edge 3) 4 [4093] 4093 innerTail561_7
    (if 3 ≤ innerCandidates561_6.length then search edge 3 [4093, 6991] innerCandidates561_6 11084 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan561_7

def innerTail561_5 : List Nat := 6823 :: innerTail561_6
def innerCandidates561_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates561_5_checked : candidates edge 4 4093 6823 innerTail561_6 best.bound = innerCandidates561_5 := by
  unfold candidates
  have row : edge 6823 = SearchData.row6823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan561_5 : scan edge (search edge 3) 4 [4093] 4093 innerTail561_5 best = best := by
  rw [innerTail561_5, scan, if_neg (by decide), innerCandidates561_5_checked]
  change scan edge (search edge 3) 4 [4093] 4093 innerTail561_6
    (if 3 ≤ innerCandidates561_5.length then search edge 3 [4093, 6823] innerCandidates561_5 10916 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan561_6

def innerTail561_4 : List Nat := 5749 :: innerTail561_5
def innerCandidates561_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates561_4_checked : candidates edge 4 4093 5749 innerTail561_5 best.bound = innerCandidates561_4 := by
  unfold candidates
  have row : edge 5749 = SearchData.row5749.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan561_4 : scan edge (search edge 3) 4 [4093] 4093 innerTail561_4 best = best := by
  rw [innerTail561_4, scan, if_neg (by decide), innerCandidates561_4_checked]
  change scan edge (search edge 3) 4 [4093] 4093 innerTail561_5
    (if 3 ≤ innerCandidates561_4.length then search edge 3 [4093, 5749] innerCandidates561_4 9842 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan561_5

def innerTail561_3 : List Nat := 5659 :: innerTail561_4
def innerCandidates561_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates561_3_checked : candidates edge 4 4093 5659 innerTail561_4 best.bound = innerCandidates561_3 := by
  unfold candidates
  have row : edge 5659 = SearchData.row5659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan561_3 : scan edge (search edge 3) 4 [4093] 4093 innerTail561_3 best = best := by
  rw [innerTail561_3, scan, if_neg (by decide), innerCandidates561_3_checked]
  change scan edge (search edge 3) 4 [4093] 4093 innerTail561_4
    (if 3 ≤ innerCandidates561_3.length then search edge 3 [4093, 5659] innerCandidates561_3 9752 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan561_4

def innerTail561_2 : List Nat := 5119 :: innerTail561_3
def innerCandidates561_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates561_2_checked : candidates edge 4 4093 5119 innerTail561_3 best.bound = innerCandidates561_2 := by
  unfold candidates
  have row : edge 5119 = SearchData.row5119.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan561_2 : scan edge (search edge 3) 4 [4093] 4093 innerTail561_2 best = best := by
  rw [innerTail561_2, scan, if_neg (by decide), innerCandidates561_2_checked]
  change scan edge (search edge 3) 4 [4093] 4093 innerTail561_3
    (if 3 ≤ innerCandidates561_2.length then search edge 3 [4093, 5119] innerCandidates561_2 9212 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan561_3

def innerTail561_1 : List Nat := 5077 :: innerTail561_2
def innerCandidates561_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates561_1_checked : candidates edge 4 4093 5077 innerTail561_2 best.bound = innerCandidates561_1 := by
  unfold candidates
  have row : edge 5077 = SearchData.row5077.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan561_1 : scan edge (search edge 3) 4 [4093] 4093 innerTail561_1 best = best := by
  rw [innerTail561_1, scan, if_neg (by decide), innerCandidates561_1_checked]
  change scan edge (search edge 3) 4 [4093] 4093 innerTail561_2
    (if 3 ≤ innerCandidates561_1.length then search edge 3 [4093, 5077] innerCandidates561_1 9170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan561_2

def innerTail561_0 : List Nat := 4567 :: innerTail561_1
def innerCandidates561_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates561_0_checked : candidates edge 4 4093 4567 innerTail561_1 best.bound = innerCandidates561_0 := by
  unfold candidates
  have row : edge 4567 = SearchData.row4567.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan561_0 : scan edge (search edge 3) 4 [4093] 4093 innerTail561_0 best = best := by
  rw [innerTail561_0, scan, if_neg (by decide), innerCandidates561_0_checked]
  change scan edge (search edge 3) 4 [4093] 4093 innerTail561_1
    (if 3 ≤ innerCandidates561_0.length then search edge 3 [4093, 4567] innerCandidates561_0 8660 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan561_1

theorem rootBranch561 : search edge 4 [4093] rootCandidates561 4093 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4093] 4093 innerTail561_0 best = best
  exact innerScan561_0

def rootCandidates560 : List Nat := [4241, 4283, 4919, 5669, 6173, 7103, 7559, 7577, 7937, 8237]
set_option maxRecDepth 4096 in
theorem rootCandidates560_checked : candidates edge 5 0 4091 tail561 best.bound = rootCandidates560 := by
  unfold candidates
  have row : edge 4091 = SearchData.row4091.contains := by funext q; rfl
  rw [row]
  decide

def innerTail560_10 : List Nat := []
theorem innerScan560_10 : scan edge (search edge 3) 4 [4091] 4091 innerTail560_10 best = best := rfl

def innerTail560_9 : List Nat := 8237 :: innerTail560_10
def innerCandidates560_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates560_9_checked : candidates edge 4 4091 8237 innerTail560_10 best.bound = innerCandidates560_9 := by
  unfold candidates
  have row : edge 8237 = SearchData.row8237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan560_9 : scan edge (search edge 3) 4 [4091] 4091 innerTail560_9 best = best := by
  rw [innerTail560_9, scan, if_neg (by decide), innerCandidates560_9_checked]
  change scan edge (search edge 3) 4 [4091] 4091 innerTail560_10
    (if 3 ≤ innerCandidates560_9.length then search edge 3 [4091, 8237] innerCandidates560_9 12328 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan560_10

def innerTail560_8 : List Nat := 7937 :: innerTail560_9
def innerCandidates560_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates560_8_checked : candidates edge 4 4091 7937 innerTail560_9 best.bound = innerCandidates560_8 := by
  unfold candidates
  have row : edge 7937 = SearchData.row7937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan560_8 : scan edge (search edge 3) 4 [4091] 4091 innerTail560_8 best = best := by
  rw [innerTail560_8, scan, if_neg (by decide), innerCandidates560_8_checked]
  change scan edge (search edge 3) 4 [4091] 4091 innerTail560_9
    (if 3 ≤ innerCandidates560_8.length then search edge 3 [4091, 7937] innerCandidates560_8 12028 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan560_9

def innerTail560_7 : List Nat := 7577 :: innerTail560_8
def innerCandidates560_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates560_7_checked : candidates edge 4 4091 7577 innerTail560_8 best.bound = innerCandidates560_7 := by
  unfold candidates
  have row : edge 7577 = SearchData.row7577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan560_7 : scan edge (search edge 3) 4 [4091] 4091 innerTail560_7 best = best := by
  rw [innerTail560_7, scan, if_neg (by decide), innerCandidates560_7_checked]
  change scan edge (search edge 3) 4 [4091] 4091 innerTail560_8
    (if 3 ≤ innerCandidates560_7.length then search edge 3 [4091, 7577] innerCandidates560_7 11668 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan560_8

def innerTail560_6 : List Nat := 7559 :: innerTail560_7
def innerCandidates560_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates560_6_checked : candidates edge 4 4091 7559 innerTail560_7 best.bound = innerCandidates560_6 := by
  unfold candidates
  have row : edge 7559 = SearchData.row7559.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan560_6 : scan edge (search edge 3) 4 [4091] 4091 innerTail560_6 best = best := by
  rw [innerTail560_6, scan, if_neg (by decide), innerCandidates560_6_checked]
  change scan edge (search edge 3) 4 [4091] 4091 innerTail560_7
    (if 3 ≤ innerCandidates560_6.length then search edge 3 [4091, 7559] innerCandidates560_6 11650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan560_7

def innerTail560_5 : List Nat := 7103 :: innerTail560_6
def innerCandidates560_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates560_5_checked : candidates edge 4 4091 7103 innerTail560_6 best.bound = innerCandidates560_5 := by
  unfold candidates
  have row : edge 7103 = SearchData.row7103.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan560_5 : scan edge (search edge 3) 4 [4091] 4091 innerTail560_5 best = best := by
  rw [innerTail560_5, scan, if_neg (by decide), innerCandidates560_5_checked]
  change scan edge (search edge 3) 4 [4091] 4091 innerTail560_6
    (if 3 ≤ innerCandidates560_5.length then search edge 3 [4091, 7103] innerCandidates560_5 11194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan560_6

def innerTail560_4 : List Nat := 6173 :: innerTail560_5
def innerCandidates560_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates560_4_checked : candidates edge 4 4091 6173 innerTail560_5 best.bound = innerCandidates560_4 := by
  unfold candidates
  have row : edge 6173 = SearchData.row6173.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan560_4 : scan edge (search edge 3) 4 [4091] 4091 innerTail560_4 best = best := by
  rw [innerTail560_4, scan, if_neg (by decide), innerCandidates560_4_checked]
  change scan edge (search edge 3) 4 [4091] 4091 innerTail560_5
    (if 3 ≤ innerCandidates560_4.length then search edge 3 [4091, 6173] innerCandidates560_4 10264 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan560_5

def innerTail560_3 : List Nat := 5669 :: innerTail560_4
def innerCandidates560_3 : List Nat := [7559]
set_option maxRecDepth 4096 in
theorem innerCandidates560_3_checked : candidates edge 4 4091 5669 innerTail560_4 best.bound = innerCandidates560_3 := by
  unfold candidates
  have row : edge 5669 = SearchData.row5669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan560_3 : scan edge (search edge 3) 4 [4091] 4091 innerTail560_3 best = best := by
  rw [innerTail560_3, scan, if_neg (by decide), innerCandidates560_3_checked]
  change scan edge (search edge 3) 4 [4091] 4091 innerTail560_4
    (if 3 ≤ innerCandidates560_3.length then search edge 3 [4091, 5669] innerCandidates560_3 9760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan560_4

def innerTail560_2 : List Nat := 4919 :: innerTail560_3
def innerCandidates560_2 : List Nat := [7559]
set_option maxRecDepth 4096 in
theorem innerCandidates560_2_checked : candidates edge 4 4091 4919 innerTail560_3 best.bound = innerCandidates560_2 := by
  unfold candidates
  have row : edge 4919 = SearchData.row4919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan560_2 : scan edge (search edge 3) 4 [4091] 4091 innerTail560_2 best = best := by
  rw [innerTail560_2, scan, if_neg (by decide), innerCandidates560_2_checked]
  change scan edge (search edge 3) 4 [4091] 4091 innerTail560_3
    (if 3 ≤ innerCandidates560_2.length then search edge 3 [4091, 4919] innerCandidates560_2 9010 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan560_3

def innerTail560_1 : List Nat := 4283 :: innerTail560_2
def innerCandidates560_1 : List Nat := [5669, 7103]
set_option maxRecDepth 4096 in
theorem innerCandidates560_1_checked : candidates edge 4 4091 4283 innerTail560_2 best.bound = innerCandidates560_1 := by
  unfold candidates
  have row : edge 4283 = SearchData.row4283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan560_1 : scan edge (search edge 3) 4 [4091] 4091 innerTail560_1 best = best := by
  rw [innerTail560_1, scan, if_neg (by decide), innerCandidates560_1_checked]
  change scan edge (search edge 3) 4 [4091] 4091 innerTail560_2
    (if 3 ≤ innerCandidates560_1.length then search edge 3 [4091, 4283] innerCandidates560_1 8374 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan560_2

def innerTail560_0 : List Nat := 4241 :: innerTail560_1
def innerCandidates560_0 : List Nat := [7103, 7559]
set_option maxRecDepth 4096 in
theorem innerCandidates560_0_checked : candidates edge 4 4091 4241 innerTail560_1 best.bound = innerCandidates560_0 := by
  unfold candidates
  have row : edge 4241 = SearchData.row4241.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan560_0 : scan edge (search edge 3) 4 [4091] 4091 innerTail560_0 best = best := by
  rw [innerTail560_0, scan, if_neg (by decide), innerCandidates560_0_checked]
  change scan edge (search edge 3) 4 [4091] 4091 innerTail560_1
    (if 3 ≤ innerCandidates560_0.length then search edge 3 [4091, 4241] innerCandidates560_0 8332 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan560_1

theorem rootBranch560 : search edge 4 [4091] rootCandidates560 4091 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4091] 4091 innerTail560_0 best = best
  exact innerScan560_0

def rootCandidates559 : List Nat := [4973, 5231, 5309, 5441, 5879, 5897, 5927, 6029, 6053, 6131, 6863]
set_option maxRecDepth 4096 in
theorem rootCandidates559_checked : candidates edge 5 0 4079 tail560 best.bound = rootCandidates559 := by
  unfold candidates
  have row : edge 4079 = SearchData.row4079.contains := by funext q; rfl
  rw [row]
  decide

def innerTail559_11 : List Nat := []
theorem innerScan559_11 : scan edge (search edge 3) 4 [4079] 4079 innerTail559_11 best = best := rfl

def innerTail559_10 : List Nat := 6863 :: innerTail559_11
def innerCandidates559_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates559_10_checked : candidates edge 4 4079 6863 innerTail559_11 best.bound = innerCandidates559_10 := by
  unfold candidates
  have row : edge 6863 = SearchData.row6863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan559_10 : scan edge (search edge 3) 4 [4079] 4079 innerTail559_10 best = best := by
  rw [innerTail559_10, scan, if_neg (by decide), innerCandidates559_10_checked]
  change scan edge (search edge 3) 4 [4079] 4079 innerTail559_11
    (if 3 ≤ innerCandidates559_10.length then search edge 3 [4079, 6863] innerCandidates559_10 10942 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan559_11

def innerTail559_9 : List Nat := 6131 :: innerTail559_10
def innerCandidates559_9 : List Nat := [6863]
set_option maxRecDepth 4096 in
theorem innerCandidates559_9_checked : candidates edge 4 4079 6131 innerTail559_10 best.bound = innerCandidates559_9 := by
  unfold candidates
  have row : edge 6131 = SearchData.row6131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan559_9 : scan edge (search edge 3) 4 [4079] 4079 innerTail559_9 best = best := by
  rw [innerTail559_9, scan, if_neg (by decide), innerCandidates559_9_checked]
  change scan edge (search edge 3) 4 [4079] 4079 innerTail559_10
    (if 3 ≤ innerCandidates559_9.length then search edge 3 [4079, 6131] innerCandidates559_9 10210 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan559_10

def innerTail559_8 : List Nat := 6053 :: innerTail559_9
def innerCandidates559_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates559_8_checked : candidates edge 4 4079 6053 innerTail559_9 best.bound = innerCandidates559_8 := by
  unfold candidates
  have row : edge 6053 = SearchData.row6053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan559_8 : scan edge (search edge 3) 4 [4079] 4079 innerTail559_8 best = best := by
  rw [innerTail559_8, scan, if_neg (by decide), innerCandidates559_8_checked]
  change scan edge (search edge 3) 4 [4079] 4079 innerTail559_9
    (if 3 ≤ innerCandidates559_8.length then search edge 3 [4079, 6053] innerCandidates559_8 10132 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan559_9

def innerTail559_7 : List Nat := 6029 :: innerTail559_8
def innerCandidates559_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates559_7_checked : candidates edge 4 4079 6029 innerTail559_8 best.bound = innerCandidates559_7 := by
  unfold candidates
  have row : edge 6029 = SearchData.row6029.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan559_7 : scan edge (search edge 3) 4 [4079] 4079 innerTail559_7 best = best := by
  rw [innerTail559_7, scan, if_neg (by decide), innerCandidates559_7_checked]
  change scan edge (search edge 3) 4 [4079] 4079 innerTail559_8
    (if 3 ≤ innerCandidates559_7.length then search edge 3 [4079, 6029] innerCandidates559_7 10108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan559_8

def innerTail559_6 : List Nat := 5927 :: innerTail559_7
def innerCandidates559_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates559_6_checked : candidates edge 4 4079 5927 innerTail559_7 best.bound = innerCandidates559_6 := by
  unfold candidates
  have row : edge 5927 = SearchData.row5927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan559_6 : scan edge (search edge 3) 4 [4079] 4079 innerTail559_6 best = best := by
  rw [innerTail559_6, scan, if_neg (by decide), innerCandidates559_6_checked]
  change scan edge (search edge 3) 4 [4079] 4079 innerTail559_7
    (if 3 ≤ innerCandidates559_6.length then search edge 3 [4079, 5927] innerCandidates559_6 10006 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan559_7

def innerTail559_5 : List Nat := 5897 :: innerTail559_6
def innerCandidates559_5 : List Nat := [6131]
set_option maxRecDepth 4096 in
theorem innerCandidates559_5_checked : candidates edge 4 4079 5897 innerTail559_6 best.bound = innerCandidates559_5 := by
  unfold candidates
  have row : edge 5897 = SearchData.row5897.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan559_5 : scan edge (search edge 3) 4 [4079] 4079 innerTail559_5 best = best := by
  rw [innerTail559_5, scan, if_neg (by decide), innerCandidates559_5_checked]
  change scan edge (search edge 3) 4 [4079] 4079 innerTail559_6
    (if 3 ≤ innerCandidates559_5.length then search edge 3 [4079, 5897] innerCandidates559_5 9976 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan559_6

def innerTail559_4 : List Nat := 5879 :: innerTail559_5
def innerCandidates559_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates559_4_checked : candidates edge 4 4079 5879 innerTail559_5 best.bound = innerCandidates559_4 := by
  unfold candidates
  have row : edge 5879 = SearchData.row5879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan559_4 : scan edge (search edge 3) 4 [4079] 4079 innerTail559_4 best = best := by
  rw [innerTail559_4, scan, if_neg (by decide), innerCandidates559_4_checked]
  change scan edge (search edge 3) 4 [4079] 4079 innerTail559_5
    (if 3 ≤ innerCandidates559_4.length then search edge 3 [4079, 5879] innerCandidates559_4 9958 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan559_5

def innerTail559_3 : List Nat := 5441 :: innerTail559_4
def innerCandidates559_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates559_3_checked : candidates edge 4 4079 5441 innerTail559_4 best.bound = innerCandidates559_3 := by
  unfold candidates
  have row : edge 5441 = SearchData.row5441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan559_3 : scan edge (search edge 3) 4 [4079] 4079 innerTail559_3 best = best := by
  rw [innerTail559_3, scan, if_neg (by decide), innerCandidates559_3_checked]
  change scan edge (search edge 3) 4 [4079] 4079 innerTail559_4
    (if 3 ≤ innerCandidates559_3.length then search edge 3 [4079, 5441] innerCandidates559_3 9520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan559_4

def innerTail559_2 : List Nat := 5309 :: innerTail559_3
def innerCandidates559_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates559_2_checked : candidates edge 4 4079 5309 innerTail559_3 best.bound = innerCandidates559_2 := by
  unfold candidates
  have row : edge 5309 = SearchData.row5309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan559_2 : scan edge (search edge 3) 4 [4079] 4079 innerTail559_2 best = best := by
  rw [innerTail559_2, scan, if_neg (by decide), innerCandidates559_2_checked]
  change scan edge (search edge 3) 4 [4079] 4079 innerTail559_3
    (if 3 ≤ innerCandidates559_2.length then search edge 3 [4079, 5309] innerCandidates559_2 9388 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan559_3

def innerTail559_1 : List Nat := 5231 :: innerTail559_2
def innerCandidates559_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates559_1_checked : candidates edge 4 4079 5231 innerTail559_2 best.bound = innerCandidates559_1 := by
  unfold candidates
  have row : edge 5231 = SearchData.row5231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan559_1 : scan edge (search edge 3) 4 [4079] 4079 innerTail559_1 best = best := by
  rw [innerTail559_1, scan, if_neg (by decide), innerCandidates559_1_checked]
  change scan edge (search edge 3) 4 [4079] 4079 innerTail559_2
    (if 3 ≤ innerCandidates559_1.length then search edge 3 [4079, 5231] innerCandidates559_1 9310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan559_2

def innerTail559_0 : List Nat := 4973 :: innerTail559_1
def innerCandidates559_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates559_0_checked : candidates edge 4 4079 4973 innerTail559_1 best.bound = innerCandidates559_0 := by
  unfold candidates
  have row : edge 4973 = SearchData.row4973.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan559_0 : scan edge (search edge 3) 4 [4079] 4079 innerTail559_0 best = best := by
  rw [innerTail559_0, scan, if_neg (by decide), innerCandidates559_0_checked]
  change scan edge (search edge 3) 4 [4079] 4079 innerTail559_1
    (if 3 ≤ innerCandidates559_0.length then search edge 3 [4079, 4973] innerCandidates559_0 9052 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan559_1

theorem rootBranch559 : search edge 4 [4079] rootCandidates559 4079 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4079] 4079 innerTail559_0 best = best
  exact innerScan559_0

def rootCandidates558 : List Nat := [4481, 4877, 5003, 5087, 5393, 6389, 6977, 7901]
set_option maxRecDepth 4096 in
theorem rootCandidates558_checked : candidates edge 5 0 4073 tail559 best.bound = rootCandidates558 := by
  unfold candidates
  have row : edge 4073 = SearchData.row4073.contains := by funext q; rfl
  rw [row]
  decide

def innerTail558_8 : List Nat := []
theorem innerScan558_8 : scan edge (search edge 3) 4 [4073] 4073 innerTail558_8 best = best := rfl

def innerTail558_7 : List Nat := 7901 :: innerTail558_8
def innerCandidates558_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates558_7_checked : candidates edge 4 4073 7901 innerTail558_8 best.bound = innerCandidates558_7 := by
  unfold candidates
  have row : edge 7901 = SearchData.row7901.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan558_7 : scan edge (search edge 3) 4 [4073] 4073 innerTail558_7 best = best := by
  rw [innerTail558_7, scan, if_neg (by decide), innerCandidates558_7_checked]
  change scan edge (search edge 3) 4 [4073] 4073 innerTail558_8
    (if 3 ≤ innerCandidates558_7.length then search edge 3 [4073, 7901] innerCandidates558_7 11974 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan558_8

def innerTail558_6 : List Nat := 6977 :: innerTail558_7
def innerCandidates558_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates558_6_checked : candidates edge 4 4073 6977 innerTail558_7 best.bound = innerCandidates558_6 := by
  unfold candidates
  have row : edge 6977 = SearchData.row6977.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan558_6 : scan edge (search edge 3) 4 [4073] 4073 innerTail558_6 best = best := by
  rw [innerTail558_6, scan, if_neg (by decide), innerCandidates558_6_checked]
  change scan edge (search edge 3) 4 [4073] 4073 innerTail558_7
    (if 3 ≤ innerCandidates558_6.length then search edge 3 [4073, 6977] innerCandidates558_6 11050 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan558_7

def innerTail558_5 : List Nat := 6389 :: innerTail558_6
def innerCandidates558_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates558_5_checked : candidates edge 4 4073 6389 innerTail558_6 best.bound = innerCandidates558_5 := by
  unfold candidates
  have row : edge 6389 = SearchData.row6389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan558_5 : scan edge (search edge 3) 4 [4073] 4073 innerTail558_5 best = best := by
  rw [innerTail558_5, scan, if_neg (by decide), innerCandidates558_5_checked]
  change scan edge (search edge 3) 4 [4073] 4073 innerTail558_6
    (if 3 ≤ innerCandidates558_5.length then search edge 3 [4073, 6389] innerCandidates558_5 10462 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan558_6

def innerTail558_4 : List Nat := 5393 :: innerTail558_5
def innerCandidates558_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates558_4_checked : candidates edge 4 4073 5393 innerTail558_5 best.bound = innerCandidates558_4 := by
  unfold candidates
  have row : edge 5393 = SearchData.row5393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan558_4 : scan edge (search edge 3) 4 [4073] 4073 innerTail558_4 best = best := by
  rw [innerTail558_4, scan, if_neg (by decide), innerCandidates558_4_checked]
  change scan edge (search edge 3) 4 [4073] 4073 innerTail558_5
    (if 3 ≤ innerCandidates558_4.length then search edge 3 [4073, 5393] innerCandidates558_4 9466 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan558_5

def innerTail558_3 : List Nat := 5087 :: innerTail558_4
def innerCandidates558_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates558_3_checked : candidates edge 4 4073 5087 innerTail558_4 best.bound = innerCandidates558_3 := by
  unfold candidates
  have row : edge 5087 = SearchData.row5087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan558_3 : scan edge (search edge 3) 4 [4073] 4073 innerTail558_3 best = best := by
  rw [innerTail558_3, scan, if_neg (by decide), innerCandidates558_3_checked]
  change scan edge (search edge 3) 4 [4073] 4073 innerTail558_4
    (if 3 ≤ innerCandidates558_3.length then search edge 3 [4073, 5087] innerCandidates558_3 9160 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan558_4

def innerTail558_2 : List Nat := 5003 :: innerTail558_3
def innerCandidates558_2 : List Nat := [5087]
set_option maxRecDepth 4096 in
theorem innerCandidates558_2_checked : candidates edge 4 4073 5003 innerTail558_3 best.bound = innerCandidates558_2 := by
  unfold candidates
  have row : edge 5003 = SearchData.row5003.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan558_2 : scan edge (search edge 3) 4 [4073] 4073 innerTail558_2 best = best := by
  rw [innerTail558_2, scan, if_neg (by decide), innerCandidates558_2_checked]
  change scan edge (search edge 3) 4 [4073] 4073 innerTail558_3
    (if 3 ≤ innerCandidates558_2.length then search edge 3 [4073, 5003] innerCandidates558_2 9076 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan558_3

def innerTail558_1 : List Nat := 4877 :: innerTail558_2
def innerCandidates558_1 : List Nat := [6977]
set_option maxRecDepth 4096 in
theorem innerCandidates558_1_checked : candidates edge 4 4073 4877 innerTail558_2 best.bound = innerCandidates558_1 := by
  unfold candidates
  have row : edge 4877 = SearchData.row4877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan558_1 : scan edge (search edge 3) 4 [4073] 4073 innerTail558_1 best = best := by
  rw [innerTail558_1, scan, if_neg (by decide), innerCandidates558_1_checked]
  change scan edge (search edge 3) 4 [4073] 4073 innerTail558_2
    (if 3 ≤ innerCandidates558_1.length then search edge 3 [4073, 4877] innerCandidates558_1 8950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan558_2

def innerTail558_0 : List Nat := 4481 :: innerTail558_1
def innerCandidates558_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates558_0_checked : candidates edge 4 4073 4481 innerTail558_1 best.bound = innerCandidates558_0 := by
  unfold candidates
  have row : edge 4481 = SearchData.row4481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan558_0 : scan edge (search edge 3) 4 [4073] 4073 innerTail558_0 best = best := by
  rw [innerTail558_0, scan, if_neg (by decide), innerCandidates558_0_checked]
  change scan edge (search edge 3) 4 [4073] 4073 innerTail558_1
    (if 3 ≤ innerCandidates558_0.length then search edge 3 [4073, 4481] innerCandidates558_0 8554 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan558_1

theorem rootBranch558 : search edge 4 [4073] rootCandidates558 4073 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4073] 4073 innerTail558_0 best = best
  exact innerScan558_0

def rootCandidates557 : List Nat := [4483, 4663, 5077, 5419, 6007, 6199, 6421, 6469, 7213, 7393, 7753, 7879]
set_option maxRecDepth 4096 in
theorem rootCandidates557_checked : candidates edge 5 0 4057 tail558 best.bound = rootCandidates557 := by
  unfold candidates
  have row : edge 4057 = SearchData.row4057.contains := by funext q; rfl
  rw [row]
  decide

def innerTail557_12 : List Nat := []
theorem innerScan557_12 : scan edge (search edge 3) 4 [4057] 4057 innerTail557_12 best = best := rfl

def innerTail557_11 : List Nat := 7879 :: innerTail557_12
def innerCandidates557_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates557_11_checked : candidates edge 4 4057 7879 innerTail557_12 best.bound = innerCandidates557_11 := by
  unfold candidates
  have row : edge 7879 = SearchData.row7879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan557_11 : scan edge (search edge 3) 4 [4057] 4057 innerTail557_11 best = best := by
  rw [innerTail557_11, scan, if_neg (by decide), innerCandidates557_11_checked]
  change scan edge (search edge 3) 4 [4057] 4057 innerTail557_12
    (if 3 ≤ innerCandidates557_11.length then search edge 3 [4057, 7879] innerCandidates557_11 11936 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan557_12

def innerTail557_10 : List Nat := 7753 :: innerTail557_11
def innerCandidates557_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates557_10_checked : candidates edge 4 4057 7753 innerTail557_11 best.bound = innerCandidates557_10 := by
  unfold candidates
  have row : edge 7753 = SearchData.row7753.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan557_10 : scan edge (search edge 3) 4 [4057] 4057 innerTail557_10 best = best := by
  rw [innerTail557_10, scan, if_neg (by decide), innerCandidates557_10_checked]
  change scan edge (search edge 3) 4 [4057] 4057 innerTail557_11
    (if 3 ≤ innerCandidates557_10.length then search edge 3 [4057, 7753] innerCandidates557_10 11810 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan557_11

def innerTail557_9 : List Nat := 7393 :: innerTail557_10
def innerCandidates557_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates557_9_checked : candidates edge 4 4057 7393 innerTail557_10 best.bound = innerCandidates557_9 := by
  unfold candidates
  have row : edge 7393 = SearchData.row7393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan557_9 : scan edge (search edge 3) 4 [4057] 4057 innerTail557_9 best = best := by
  rw [innerTail557_9, scan, if_neg (by decide), innerCandidates557_9_checked]
  change scan edge (search edge 3) 4 [4057] 4057 innerTail557_10
    (if 3 ≤ innerCandidates557_9.length then search edge 3 [4057, 7393] innerCandidates557_9 11450 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan557_10

def innerTail557_8 : List Nat := 7213 :: innerTail557_9
def innerCandidates557_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates557_8_checked : candidates edge 4 4057 7213 innerTail557_9 best.bound = innerCandidates557_8 := by
  unfold candidates
  have row : edge 7213 = SearchData.row7213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan557_8 : scan edge (search edge 3) 4 [4057] 4057 innerTail557_8 best = best := by
  rw [innerTail557_8, scan, if_neg (by decide), innerCandidates557_8_checked]
  change scan edge (search edge 3) 4 [4057] 4057 innerTail557_9
    (if 3 ≤ innerCandidates557_8.length then search edge 3 [4057, 7213] innerCandidates557_8 11270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan557_9

def innerTail557_7 : List Nat := 6469 :: innerTail557_8
def innerCandidates557_7 : List Nat := [7879]
set_option maxRecDepth 4096 in
theorem innerCandidates557_7_checked : candidates edge 4 4057 6469 innerTail557_8 best.bound = innerCandidates557_7 := by
  unfold candidates
  have row : edge 6469 = SearchData.row6469.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan557_7 : scan edge (search edge 3) 4 [4057] 4057 innerTail557_7 best = best := by
  rw [innerTail557_7, scan, if_neg (by decide), innerCandidates557_7_checked]
  change scan edge (search edge 3) 4 [4057] 4057 innerTail557_8
    (if 3 ≤ innerCandidates557_7.length then search edge 3 [4057, 6469] innerCandidates557_7 10526 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan557_8

def innerTail557_6 : List Nat := 6421 :: innerTail557_7
def innerCandidates557_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates557_6_checked : candidates edge 4 4057 6421 innerTail557_7 best.bound = innerCandidates557_6 := by
  unfold candidates
  have row : edge 6421 = SearchData.row6421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan557_6 : scan edge (search edge 3) 4 [4057] 4057 innerTail557_6 best = best := by
  rw [innerTail557_6, scan, if_neg (by decide), innerCandidates557_6_checked]
  change scan edge (search edge 3) 4 [4057] 4057 innerTail557_7
    (if 3 ≤ innerCandidates557_6.length then search edge 3 [4057, 6421] innerCandidates557_6 10478 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan557_7

def innerTail557_5 : List Nat := 6199 :: innerTail557_6
def innerCandidates557_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates557_5_checked : candidates edge 4 4057 6199 innerTail557_6 best.bound = innerCandidates557_5 := by
  unfold candidates
  have row : edge 6199 = SearchData.row6199.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan557_5 : scan edge (search edge 3) 4 [4057] 4057 innerTail557_5 best = best := by
  rw [innerTail557_5, scan, if_neg (by decide), innerCandidates557_5_checked]
  change scan edge (search edge 3) 4 [4057] 4057 innerTail557_6
    (if 3 ≤ innerCandidates557_5.length then search edge 3 [4057, 6199] innerCandidates557_5 10256 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan557_6

def innerTail557_4 : List Nat := 6007 :: innerTail557_5
def innerCandidates557_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates557_4_checked : candidates edge 4 4057 6007 innerTail557_5 best.bound = innerCandidates557_4 := by
  unfold candidates
  have row : edge 6007 = SearchData.row6007.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan557_4 : scan edge (search edge 3) 4 [4057] 4057 innerTail557_4 best = best := by
  rw [innerTail557_4, scan, if_neg (by decide), innerCandidates557_4_checked]
  change scan edge (search edge 3) 4 [4057] 4057 innerTail557_5
    (if 3 ≤ innerCandidates557_4.length then search edge 3 [4057, 6007] innerCandidates557_4 10064 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan557_5

def innerTail557_3 : List Nat := 5419 :: innerTail557_4
def innerCandidates557_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates557_3_checked : candidates edge 4 4057 5419 innerTail557_4 best.bound = innerCandidates557_3 := by
  unfold candidates
  have row : edge 5419 = SearchData.row5419.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan557_3 : scan edge (search edge 3) 4 [4057] 4057 innerTail557_3 best = best := by
  rw [innerTail557_3, scan, if_neg (by decide), innerCandidates557_3_checked]
  change scan edge (search edge 3) 4 [4057] 4057 innerTail557_4
    (if 3 ≤ innerCandidates557_3.length then search edge 3 [4057, 5419] innerCandidates557_3 9476 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan557_4

def innerTail557_2 : List Nat := 5077 :: innerTail557_3
def innerCandidates557_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates557_2_checked : candidates edge 4 4057 5077 innerTail557_3 best.bound = innerCandidates557_2 := by
  unfold candidates
  have row : edge 5077 = SearchData.row5077.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan557_2 : scan edge (search edge 3) 4 [4057] 4057 innerTail557_2 best = best := by
  rw [innerTail557_2, scan, if_neg (by decide), innerCandidates557_2_checked]
  change scan edge (search edge 3) 4 [4057] 4057 innerTail557_3
    (if 3 ≤ innerCandidates557_2.length then search edge 3 [4057, 5077] innerCandidates557_2 9134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan557_3

def innerTail557_1 : List Nat := 4663 :: innerTail557_2
def innerCandidates557_1 : List Nat := [7753, 7879]
set_option maxRecDepth 4096 in
theorem innerCandidates557_1_checked : candidates edge 4 4057 4663 innerTail557_2 best.bound = innerCandidates557_1 := by
  unfold candidates
  have row : edge 4663 = SearchData.row4663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan557_1 : scan edge (search edge 3) 4 [4057] 4057 innerTail557_1 best = best := by
  rw [innerTail557_1, scan, if_neg (by decide), innerCandidates557_1_checked]
  change scan edge (search edge 3) 4 [4057] 4057 innerTail557_2
    (if 3 ≤ innerCandidates557_1.length then search edge 3 [4057, 4663] innerCandidates557_1 8720 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan557_2

def innerTail557_0 : List Nat := 4483 :: innerTail557_1
def innerCandidates557_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates557_0_checked : candidates edge 4 4057 4483 innerTail557_1 best.bound = innerCandidates557_0 := by
  unfold candidates
  have row : edge 4483 = SearchData.row4483.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan557_0 : scan edge (search edge 3) 4 [4057] 4057 innerTail557_0 best = best := by
  rw [innerTail557_0, scan, if_neg (by decide), innerCandidates557_0_checked]
  change scan edge (search edge 3) 4 [4057] 4057 innerTail557_1
    (if 3 ≤ innerCandidates557_0.length then search edge 3 [4057, 4483] innerCandidates557_0 8540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan557_1

theorem rootBranch557 : search edge 4 [4057] rootCandidates557 4057 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4057] 4057 innerTail557_0 best = best
  exact innerScan557_0

end ProjectEulerSolutions.PrimePairUpper.Search
