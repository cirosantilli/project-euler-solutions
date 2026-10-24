import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates831 : List Nat := [6793, 7213, 7321, 8179]
set_option maxRecDepth 4096 in
theorem rootCandidates831_checked : candidates edge 5 0 6397 tail832 best.bound = rootCandidates831 := by
  unfold candidates
  have row : edge 6397 = SearchData.row6397.contains := by funext q; rfl
  rw [row]
  decide

def innerTail831_4 : List Nat := []
theorem innerScan831_4 : scan edge (search edge 3) 4 [6397] 6397 innerTail831_4 best = best := rfl

def innerTail831_3 : List Nat := 8179 :: innerTail831_4
def innerCandidates831_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates831_3_checked : candidates edge 4 6397 8179 innerTail831_4 best.bound = innerCandidates831_3 := by
  unfold candidates
  have row : edge 8179 = SearchData.row8179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan831_3 : scan edge (search edge 3) 4 [6397] 6397 innerTail831_3 best = best := by
  rw [innerTail831_3, scan, if_neg (by decide), innerCandidates831_3_checked]
  change scan edge (search edge 3) 4 [6397] 6397 innerTail831_4
    (if 3 ≤ innerCandidates831_3.length then search edge 3 [6397, 8179] innerCandidates831_3 14576 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan831_4

def innerTail831_2 : List Nat := 7321 :: innerTail831_3
def innerCandidates831_2 : List Nat := [8179]
set_option maxRecDepth 4096 in
theorem innerCandidates831_2_checked : candidates edge 4 6397 7321 innerTail831_3 best.bound = innerCandidates831_2 := by
  unfold candidates
  have row : edge 7321 = SearchData.row7321.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan831_2 : scan edge (search edge 3) 4 [6397] 6397 innerTail831_2 best = best := by
  rw [innerTail831_2, scan, if_neg (by decide), innerCandidates831_2_checked]
  change scan edge (search edge 3) 4 [6397] 6397 innerTail831_3
    (if 3 ≤ innerCandidates831_2.length then search edge 3 [6397, 7321] innerCandidates831_2 13718 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan831_3

def innerTail831_1 : List Nat := 7213 :: innerTail831_2
def innerCandidates831_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates831_1_checked : candidates edge 4 6397 7213 innerTail831_2 best.bound = innerCandidates831_1 := by
  unfold candidates
  have row : edge 7213 = SearchData.row7213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan831_1 : scan edge (search edge 3) 4 [6397] 6397 innerTail831_1 best = best := by
  rw [innerTail831_1, scan, if_neg (by decide), innerCandidates831_1_checked]
  change scan edge (search edge 3) 4 [6397] 6397 innerTail831_2
    (if 3 ≤ innerCandidates831_1.length then search edge 3 [6397, 7213] innerCandidates831_1 13610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan831_2

def innerTail831_0 : List Nat := 6793 :: innerTail831_1
def innerCandidates831_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates831_0_checked : candidates edge 4 6397 6793 innerTail831_1 best.bound = innerCandidates831_0 := by
  unfold candidates
  have row : edge 6793 = SearchData.row6793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan831_0 : scan edge (search edge 3) 4 [6397] 6397 innerTail831_0 best = best := by
  rw [innerTail831_0, scan, if_neg (by decide), innerCandidates831_0_checked]
  change scan edge (search edge 3) 4 [6397] 6397 innerTail831_1
    (if 3 ≤ innerCandidates831_0.length then search edge 3 [6397, 6793] innerCandidates831_0 13190 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan831_1

theorem rootBranch831 : search edge 4 [6397] rootCandidates831 6397 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6397] 6397 innerTail831_0 best = best
  exact innerScan831_0

def rootCandidates830 : List Nat := [6473, 7481, 7523, 7937, 8111, 8363]
set_option maxRecDepth 4096 in
theorem rootCandidates830_checked : candidates edge 5 0 6389 tail831 best.bound = rootCandidates830 := by
  unfold candidates
  have row : edge 6389 = SearchData.row6389.contains := by funext q; rfl
  rw [row]
  decide

def innerTail830_6 : List Nat := []
theorem innerScan830_6 : scan edge (search edge 3) 4 [6389] 6389 innerTail830_6 best = best := rfl

def innerTail830_5 : List Nat := 8363 :: innerTail830_6
def innerCandidates830_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates830_5_checked : candidates edge 4 6389 8363 innerTail830_6 best.bound = innerCandidates830_5 := by
  unfold candidates
  have row : edge 8363 = SearchData.row8363.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan830_5 : scan edge (search edge 3) 4 [6389] 6389 innerTail830_5 best = best := by
  rw [innerTail830_5, scan, if_neg (by decide), innerCandidates830_5_checked]
  change scan edge (search edge 3) 4 [6389] 6389 innerTail830_6
    (if 3 ≤ innerCandidates830_5.length then search edge 3 [6389, 8363] innerCandidates830_5 14752 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan830_6

def innerTail830_4 : List Nat := 8111 :: innerTail830_5
def innerCandidates830_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates830_4_checked : candidates edge 4 6389 8111 innerTail830_5 best.bound = innerCandidates830_4 := by
  unfold candidates
  have row : edge 8111 = SearchData.row8111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan830_4 : scan edge (search edge 3) 4 [6389] 6389 innerTail830_4 best = best := by
  rw [innerTail830_4, scan, if_neg (by decide), innerCandidates830_4_checked]
  change scan edge (search edge 3) 4 [6389] 6389 innerTail830_5
    (if 3 ≤ innerCandidates830_4.length then search edge 3 [6389, 8111] innerCandidates830_4 14500 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan830_5

def innerTail830_3 : List Nat := 7937 :: innerTail830_4
def innerCandidates830_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates830_3_checked : candidates edge 4 6389 7937 innerTail830_4 best.bound = innerCandidates830_3 := by
  unfold candidates
  have row : edge 7937 = SearchData.row7937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan830_3 : scan edge (search edge 3) 4 [6389] 6389 innerTail830_3 best = best := by
  rw [innerTail830_3, scan, if_neg (by decide), innerCandidates830_3_checked]
  change scan edge (search edge 3) 4 [6389] 6389 innerTail830_4
    (if 3 ≤ innerCandidates830_3.length then search edge 3 [6389, 7937] innerCandidates830_3 14326 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan830_4

def innerTail830_2 : List Nat := 7523 :: innerTail830_3
def innerCandidates830_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates830_2_checked : candidates edge 4 6389 7523 innerTail830_3 best.bound = innerCandidates830_2 := by
  unfold candidates
  have row : edge 7523 = SearchData.row7523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan830_2 : scan edge (search edge 3) 4 [6389] 6389 innerTail830_2 best = best := by
  rw [innerTail830_2, scan, if_neg (by decide), innerCandidates830_2_checked]
  change scan edge (search edge 3) 4 [6389] 6389 innerTail830_3
    (if 3 ≤ innerCandidates830_2.length then search edge 3 [6389, 7523] innerCandidates830_2 13912 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan830_3

def innerTail830_1 : List Nat := 7481 :: innerTail830_2
def innerCandidates830_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates830_1_checked : candidates edge 4 6389 7481 innerTail830_2 best.bound = innerCandidates830_1 := by
  unfold candidates
  have row : edge 7481 = SearchData.row7481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan830_1 : scan edge (search edge 3) 4 [6389] 6389 innerTail830_1 best = best := by
  rw [innerTail830_1, scan, if_neg (by decide), innerCandidates830_1_checked]
  change scan edge (search edge 3) 4 [6389] 6389 innerTail830_2
    (if 3 ≤ innerCandidates830_1.length then search edge 3 [6389, 7481] innerCandidates830_1 13870 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan830_2

def innerTail830_0 : List Nat := 6473 :: innerTail830_1
def innerCandidates830_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates830_0_checked : candidates edge 4 6389 6473 innerTail830_1 best.bound = innerCandidates830_0 := by
  unfold candidates
  have row : edge 6473 = SearchData.row6473.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan830_0 : scan edge (search edge 3) 4 [6389] 6389 innerTail830_0 best = best := by
  rw [innerTail830_0, scan, if_neg (by decide), innerCandidates830_0_checked]
  change scan edge (search edge 3) 4 [6389] 6389 innerTail830_1
    (if 3 ≤ innerCandidates830_0.length then search edge 3 [6389, 6473] innerCandidates830_0 12862 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan830_1

theorem rootBranch830 : search edge 4 [6389] rootCandidates830 6389 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6389] 6389 innerTail830_0 best = best
  exact innerScan830_0

def rootCandidates829 : List Nat := [6703, 7057, 8017, 8233]
set_option maxRecDepth 4096 in
theorem rootCandidates829_checked : candidates edge 5 0 6379 tail830 best.bound = rootCandidates829 := by
  unfold candidates
  have row : edge 6379 = SearchData.row6379.contains := by funext q; rfl
  rw [row]
  decide

def innerTail829_4 : List Nat := []
theorem innerScan829_4 : scan edge (search edge 3) 4 [6379] 6379 innerTail829_4 best = best := rfl

def innerTail829_3 : List Nat := 8233 :: innerTail829_4
def innerCandidates829_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates829_3_checked : candidates edge 4 6379 8233 innerTail829_4 best.bound = innerCandidates829_3 := by
  unfold candidates
  have row : edge 8233 = SearchData.row8233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan829_3 : scan edge (search edge 3) 4 [6379] 6379 innerTail829_3 best = best := by
  rw [innerTail829_3, scan, if_neg (by decide), innerCandidates829_3_checked]
  change scan edge (search edge 3) 4 [6379] 6379 innerTail829_4
    (if 3 ≤ innerCandidates829_3.length then search edge 3 [6379, 8233] innerCandidates829_3 14612 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan829_4

def innerTail829_2 : List Nat := 8017 :: innerTail829_3
def innerCandidates829_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates829_2_checked : candidates edge 4 6379 8017 innerTail829_3 best.bound = innerCandidates829_2 := by
  unfold candidates
  have row : edge 8017 = SearchData.row8017.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan829_2 : scan edge (search edge 3) 4 [6379] 6379 innerTail829_2 best = best := by
  rw [innerTail829_2, scan, if_neg (by decide), innerCandidates829_2_checked]
  change scan edge (search edge 3) 4 [6379] 6379 innerTail829_3
    (if 3 ≤ innerCandidates829_2.length then search edge 3 [6379, 8017] innerCandidates829_2 14396 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan829_3

def innerTail829_1 : List Nat := 7057 :: innerTail829_2
def innerCandidates829_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates829_1_checked : candidates edge 4 6379 7057 innerTail829_2 best.bound = innerCandidates829_1 := by
  unfold candidates
  have row : edge 7057 = SearchData.row7057.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan829_1 : scan edge (search edge 3) 4 [6379] 6379 innerTail829_1 best = best := by
  rw [innerTail829_1, scan, if_neg (by decide), innerCandidates829_1_checked]
  change scan edge (search edge 3) 4 [6379] 6379 innerTail829_2
    (if 3 ≤ innerCandidates829_1.length then search edge 3 [6379, 7057] innerCandidates829_1 13436 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan829_2

def innerTail829_0 : List Nat := 6703 :: innerTail829_1
def innerCandidates829_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates829_0_checked : candidates edge 4 6379 6703 innerTail829_1 best.bound = innerCandidates829_0 := by
  unfold candidates
  have row : edge 6703 = SearchData.row6703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan829_0 : scan edge (search edge 3) 4 [6379] 6379 innerTail829_0 best = best := by
  rw [innerTail829_0, scan, if_neg (by decide), innerCandidates829_0_checked]
  change scan edge (search edge 3) 4 [6379] 6379 innerTail829_1
    (if 3 ≤ innerCandidates829_0.length then search edge 3 [6379, 6703] innerCandidates829_0 13082 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan829_1

theorem rootBranch829 : search edge 4 [6379] rootCandidates829 6379 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6379] 6379 innerTail829_0 best = best
  exact innerScan829_0

def rootCandidates828 : List Nat := [6661, 6781, 7459, 7507, 7699, 8167]
set_option maxRecDepth 4096 in
theorem rootCandidates828_checked : candidates edge 5 0 6373 tail829 best.bound = rootCandidates828 := by
  unfold candidates
  have row : edge 6373 = SearchData.row6373.contains := by funext q; rfl
  rw [row]
  decide

def innerTail828_6 : List Nat := []
theorem innerScan828_6 : scan edge (search edge 3) 4 [6373] 6373 innerTail828_6 best = best := rfl

def innerTail828_5 : List Nat := 8167 :: innerTail828_6
def innerCandidates828_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates828_5_checked : candidates edge 4 6373 8167 innerTail828_6 best.bound = innerCandidates828_5 := by
  unfold candidates
  have row : edge 8167 = SearchData.row8167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan828_5 : scan edge (search edge 3) 4 [6373] 6373 innerTail828_5 best = best := by
  rw [innerTail828_5, scan, if_neg (by decide), innerCandidates828_5_checked]
  change scan edge (search edge 3) 4 [6373] 6373 innerTail828_6
    (if 3 ≤ innerCandidates828_5.length then search edge 3 [6373, 8167] innerCandidates828_5 14540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan828_6

def innerTail828_4 : List Nat := 7699 :: innerTail828_5
def innerCandidates828_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates828_4_checked : candidates edge 4 6373 7699 innerTail828_5 best.bound = innerCandidates828_4 := by
  unfold candidates
  have row : edge 7699 = SearchData.row7699.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan828_4 : scan edge (search edge 3) 4 [6373] 6373 innerTail828_4 best = best := by
  rw [innerTail828_4, scan, if_neg (by decide), innerCandidates828_4_checked]
  change scan edge (search edge 3) 4 [6373] 6373 innerTail828_5
    (if 3 ≤ innerCandidates828_4.length then search edge 3 [6373, 7699] innerCandidates828_4 14072 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan828_5

def innerTail828_3 : List Nat := 7507 :: innerTail828_4
def innerCandidates828_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates828_3_checked : candidates edge 4 6373 7507 innerTail828_4 best.bound = innerCandidates828_3 := by
  unfold candidates
  have row : edge 7507 = SearchData.row7507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan828_3 : scan edge (search edge 3) 4 [6373] 6373 innerTail828_3 best = best := by
  rw [innerTail828_3, scan, if_neg (by decide), innerCandidates828_3_checked]
  change scan edge (search edge 3) 4 [6373] 6373 innerTail828_4
    (if 3 ≤ innerCandidates828_3.length then search edge 3 [6373, 7507] innerCandidates828_3 13880 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan828_4

def innerTail828_2 : List Nat := 7459 :: innerTail828_3
def innerCandidates828_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates828_2_checked : candidates edge 4 6373 7459 innerTail828_3 best.bound = innerCandidates828_2 := by
  unfold candidates
  have row : edge 7459 = SearchData.row7459.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan828_2 : scan edge (search edge 3) 4 [6373] 6373 innerTail828_2 best = best := by
  rw [innerTail828_2, scan, if_neg (by decide), innerCandidates828_2_checked]
  change scan edge (search edge 3) 4 [6373] 6373 innerTail828_3
    (if 3 ≤ innerCandidates828_2.length then search edge 3 [6373, 7459] innerCandidates828_2 13832 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan828_3

def innerTail828_1 : List Nat := 6781 :: innerTail828_2
def innerCandidates828_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates828_1_checked : candidates edge 4 6373 6781 innerTail828_2 best.bound = innerCandidates828_1 := by
  unfold candidates
  have row : edge 6781 = SearchData.row6781.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan828_1 : scan edge (search edge 3) 4 [6373] 6373 innerTail828_1 best = best := by
  rw [innerTail828_1, scan, if_neg (by decide), innerCandidates828_1_checked]
  change scan edge (search edge 3) 4 [6373] 6373 innerTail828_2
    (if 3 ≤ innerCandidates828_1.length then search edge 3 [6373, 6781] innerCandidates828_1 13154 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan828_2

def innerTail828_0 : List Nat := 6661 :: innerTail828_1
def innerCandidates828_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates828_0_checked : candidates edge 4 6373 6661 innerTail828_1 best.bound = innerCandidates828_0 := by
  unfold candidates
  have row : edge 6661 = SearchData.row6661.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan828_0 : scan edge (search edge 3) 4 [6373] 6373 innerTail828_0 best = best := by
  rw [innerTail828_0, scan, if_neg (by decide), innerCandidates828_0_checked]
  change scan edge (search edge 3) 4 [6373] 6373 innerTail828_1
    (if 3 ≤ innerCandidates828_0.length then search edge 3 [6373, 6661] innerCandidates828_0 13034 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan828_1

theorem rootBranch828 : search edge 4 [6373] rootCandidates828 6373 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6373] 6373 innerTail828_0 best = best
  exact innerScan828_0

def rootCandidates827 : List Nat := [6703, 6793, 7477, 8353]
set_option maxRecDepth 4096 in
theorem rootCandidates827_checked : candidates edge 5 0 6367 tail828 best.bound = rootCandidates827 := by
  unfold candidates
  have row : edge 6367 = SearchData.row6367.contains := by funext q; rfl
  rw [row]
  decide

def innerTail827_4 : List Nat := []
theorem innerScan827_4 : scan edge (search edge 3) 4 [6367] 6367 innerTail827_4 best = best := rfl

def innerTail827_3 : List Nat := 8353 :: innerTail827_4
def innerCandidates827_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates827_3_checked : candidates edge 4 6367 8353 innerTail827_4 best.bound = innerCandidates827_3 := by
  unfold candidates
  have row : edge 8353 = SearchData.row8353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan827_3 : scan edge (search edge 3) 4 [6367] 6367 innerTail827_3 best = best := by
  rw [innerTail827_3, scan, if_neg (by decide), innerCandidates827_3_checked]
  change scan edge (search edge 3) 4 [6367] 6367 innerTail827_4
    (if 3 ≤ innerCandidates827_3.length then search edge 3 [6367, 8353] innerCandidates827_3 14720 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan827_4

def innerTail827_2 : List Nat := 7477 :: innerTail827_3
def innerCandidates827_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates827_2_checked : candidates edge 4 6367 7477 innerTail827_3 best.bound = innerCandidates827_2 := by
  unfold candidates
  have row : edge 7477 = SearchData.row7477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan827_2 : scan edge (search edge 3) 4 [6367] 6367 innerTail827_2 best = best := by
  rw [innerTail827_2, scan, if_neg (by decide), innerCandidates827_2_checked]
  change scan edge (search edge 3) 4 [6367] 6367 innerTail827_3
    (if 3 ≤ innerCandidates827_2.length then search edge 3 [6367, 7477] innerCandidates827_2 13844 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan827_3

def innerTail827_1 : List Nat := 6793 :: innerTail827_2
def innerCandidates827_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates827_1_checked : candidates edge 4 6367 6793 innerTail827_2 best.bound = innerCandidates827_1 := by
  unfold candidates
  have row : edge 6793 = SearchData.row6793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan827_1 : scan edge (search edge 3) 4 [6367] 6367 innerTail827_1 best = best := by
  rw [innerTail827_1, scan, if_neg (by decide), innerCandidates827_1_checked]
  change scan edge (search edge 3) 4 [6367] 6367 innerTail827_2
    (if 3 ≤ innerCandidates827_1.length then search edge 3 [6367, 6793] innerCandidates827_1 13160 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan827_2

def innerTail827_0 : List Nat := 6703 :: innerTail827_1
def innerCandidates827_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates827_0_checked : candidates edge 4 6367 6703 innerTail827_1 best.bound = innerCandidates827_0 := by
  unfold candidates
  have row : edge 6703 = SearchData.row6703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan827_0 : scan edge (search edge 3) 4 [6367] 6367 innerTail827_0 best = best := by
  rw [innerTail827_0, scan, if_neg (by decide), innerCandidates827_0_checked]
  change scan edge (search edge 3) 4 [6367] 6367 innerTail827_1
    (if 3 ≤ innerCandidates827_0.length then search edge 3 [6367, 6703] innerCandidates827_0 13070 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan827_1

theorem rootBranch827 : search edge 4 [6367] rootCandidates827 6367 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6367] 6367 innerTail827_0 best = best
  exact innerScan827_0

def rootCandidates826 : List Nat := [6481, 6673, 7069, 7507]
set_option maxRecDepth 4096 in
theorem rootCandidates826_checked : candidates edge 5 0 6361 tail827 best.bound = rootCandidates826 := by
  unfold candidates
  have row : edge 6361 = SearchData.row6361.contains := by funext q; rfl
  rw [row]
  decide

def innerTail826_4 : List Nat := []
theorem innerScan826_4 : scan edge (search edge 3) 4 [6361] 6361 innerTail826_4 best = best := rfl

def innerTail826_3 : List Nat := 7507 :: innerTail826_4
def innerCandidates826_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates826_3_checked : candidates edge 4 6361 7507 innerTail826_4 best.bound = innerCandidates826_3 := by
  unfold candidates
  have row : edge 7507 = SearchData.row7507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan826_3 : scan edge (search edge 3) 4 [6361] 6361 innerTail826_3 best = best := by
  rw [innerTail826_3, scan, if_neg (by decide), innerCandidates826_3_checked]
  change scan edge (search edge 3) 4 [6361] 6361 innerTail826_4
    (if 3 ≤ innerCandidates826_3.length then search edge 3 [6361, 7507] innerCandidates826_3 13868 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan826_4

def innerTail826_2 : List Nat := 7069 :: innerTail826_3
def innerCandidates826_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates826_2_checked : candidates edge 4 6361 7069 innerTail826_3 best.bound = innerCandidates826_2 := by
  unfold candidates
  have row : edge 7069 = SearchData.row7069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan826_2 : scan edge (search edge 3) 4 [6361] 6361 innerTail826_2 best = best := by
  rw [innerTail826_2, scan, if_neg (by decide), innerCandidates826_2_checked]
  change scan edge (search edge 3) 4 [6361] 6361 innerTail826_3
    (if 3 ≤ innerCandidates826_2.length then search edge 3 [6361, 7069] innerCandidates826_2 13430 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan826_3

def innerTail826_1 : List Nat := 6673 :: innerTail826_2
def innerCandidates826_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates826_1_checked : candidates edge 4 6361 6673 innerTail826_2 best.bound = innerCandidates826_1 := by
  unfold candidates
  have row : edge 6673 = SearchData.row6673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan826_1 : scan edge (search edge 3) 4 [6361] 6361 innerTail826_1 best = best := by
  rw [innerTail826_1, scan, if_neg (by decide), innerCandidates826_1_checked]
  change scan edge (search edge 3) 4 [6361] 6361 innerTail826_2
    (if 3 ≤ innerCandidates826_1.length then search edge 3 [6361, 6673] innerCandidates826_1 13034 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan826_2

def innerTail826_0 : List Nat := 6481 :: innerTail826_1
def innerCandidates826_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates826_0_checked : candidates edge 4 6361 6481 innerTail826_1 best.bound = innerCandidates826_0 := by
  unfold candidates
  have row : edge 6481 = SearchData.row6481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan826_0 : scan edge (search edge 3) 4 [6361] 6361 innerTail826_0 best = best := by
  rw [innerTail826_0, scan, if_neg (by decide), innerCandidates826_0_checked]
  change scan edge (search edge 3) 4 [6361] 6361 innerTail826_1
    (if 3 ≤ innerCandidates826_0.length then search edge 3 [6361, 6481] innerCandidates826_0 12842 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan826_1

theorem rootBranch826 : search edge 4 [6361] rootCandidates826 6361 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6361] 6361 innerTail826_0 best = best
  exact innerScan826_0

def rootCandidates825 : List Nat := [6737, 6761, 7019, 7577, 8123]
set_option maxRecDepth 4096 in
theorem rootCandidates825_checked : candidates edge 5 0 6359 tail826 best.bound = rootCandidates825 := by
  unfold candidates
  have row : edge 6359 = SearchData.row6359.contains := by funext q; rfl
  rw [row]
  decide

def innerTail825_5 : List Nat := []
theorem innerScan825_5 : scan edge (search edge 3) 4 [6359] 6359 innerTail825_5 best = best := rfl

def innerTail825_4 : List Nat := 8123 :: innerTail825_5
def innerCandidates825_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates825_4_checked : candidates edge 4 6359 8123 innerTail825_5 best.bound = innerCandidates825_4 := by
  unfold candidates
  have row : edge 8123 = SearchData.row8123.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan825_4 : scan edge (search edge 3) 4 [6359] 6359 innerTail825_4 best = best := by
  rw [innerTail825_4, scan, if_neg (by decide), innerCandidates825_4_checked]
  change scan edge (search edge 3) 4 [6359] 6359 innerTail825_5
    (if 3 ≤ innerCandidates825_4.length then search edge 3 [6359, 8123] innerCandidates825_4 14482 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan825_5

def innerTail825_3 : List Nat := 7577 :: innerTail825_4
def innerCandidates825_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates825_3_checked : candidates edge 4 6359 7577 innerTail825_4 best.bound = innerCandidates825_3 := by
  unfold candidates
  have row : edge 7577 = SearchData.row7577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan825_3 : scan edge (search edge 3) 4 [6359] 6359 innerTail825_3 best = best := by
  rw [innerTail825_3, scan, if_neg (by decide), innerCandidates825_3_checked]
  change scan edge (search edge 3) 4 [6359] 6359 innerTail825_4
    (if 3 ≤ innerCandidates825_3.length then search edge 3 [6359, 7577] innerCandidates825_3 13936 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan825_4

def innerTail825_2 : List Nat := 7019 :: innerTail825_3
def innerCandidates825_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates825_2_checked : candidates edge 4 6359 7019 innerTail825_3 best.bound = innerCandidates825_2 := by
  unfold candidates
  have row : edge 7019 = SearchData.row7019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan825_2 : scan edge (search edge 3) 4 [6359] 6359 innerTail825_2 best = best := by
  rw [innerTail825_2, scan, if_neg (by decide), innerCandidates825_2_checked]
  change scan edge (search edge 3) 4 [6359] 6359 innerTail825_3
    (if 3 ≤ innerCandidates825_2.length then search edge 3 [6359, 7019] innerCandidates825_2 13378 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan825_3

def innerTail825_1 : List Nat := 6761 :: innerTail825_2
def innerCandidates825_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates825_1_checked : candidates edge 4 6359 6761 innerTail825_2 best.bound = innerCandidates825_1 := by
  unfold candidates
  have row : edge 6761 = SearchData.row6761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan825_1 : scan edge (search edge 3) 4 [6359] 6359 innerTail825_1 best = best := by
  rw [innerTail825_1, scan, if_neg (by decide), innerCandidates825_1_checked]
  change scan edge (search edge 3) 4 [6359] 6359 innerTail825_2
    (if 3 ≤ innerCandidates825_1.length then search edge 3 [6359, 6761] innerCandidates825_1 13120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan825_2

def innerTail825_0 : List Nat := 6737 :: innerTail825_1
def innerCandidates825_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates825_0_checked : candidates edge 4 6359 6737 innerTail825_1 best.bound = innerCandidates825_0 := by
  unfold candidates
  have row : edge 6737 = SearchData.row6737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan825_0 : scan edge (search edge 3) 4 [6359] 6359 innerTail825_0 best = best := by
  rw [innerTail825_0, scan, if_neg (by decide), innerCandidates825_0_checked]
  change scan edge (search edge 3) 4 [6359] 6359 innerTail825_1
    (if 3 ≤ innerCandidates825_0.length then search edge 3 [6359, 6737] innerCandidates825_0 13096 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan825_1

theorem rootBranch825 : search edge 4 [6359] rootCandidates825 6359 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6359] 6359 innerTail825_0 best = best
  exact innerScan825_0

def rootCandidates824 : List Nat := [6359, 7949]
set_option maxRecDepth 4096 in
theorem rootCandidates824_checked : candidates edge 5 0 6353 tail825 best.bound = rootCandidates824 := by
  unfold candidates
  have row : edge 6353 = SearchData.row6353.contains := by funext q; rfl
  rw [row]
  decide

end ProjectEulerSolutions.PrimePairUpper.Search
