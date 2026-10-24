import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates823 : List Nat := [7039, 7669, 7867]
set_option maxRecDepth 4096 in
theorem rootCandidates823_checked : candidates edge 5 0 6343 tail824 best.bound = rootCandidates823 := by
  unfold candidates
  have row : edge 6343 = SearchData.row6343.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates822 : List Nat := [6949, 6991, 7933, 8101]
set_option maxRecDepth 4096 in
theorem rootCandidates822_checked : candidates edge 5 0 6337 tail823 best.bound = rootCandidates822 := by
  unfold candidates
  have row : edge 6337 = SearchData.row6337.contains := by funext q; rfl
  rw [row]
  decide

def innerTail822_4 : List Nat := []
theorem innerScan822_4 : scan edge (search edge 3) 4 [6337] 6337 innerTail822_4 best = best := rfl

def innerTail822_3 : List Nat := 8101 :: innerTail822_4
def innerCandidates822_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates822_3_checked : candidates edge 4 6337 8101 innerTail822_4 best.bound = innerCandidates822_3 := by
  unfold candidates
  have row : edge 8101 = SearchData.row8101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan822_3 : scan edge (search edge 3) 4 [6337] 6337 innerTail822_3 best = best := by
  rw [innerTail822_3, scan, if_neg (by decide), innerCandidates822_3_checked]
  change scan edge (search edge 3) 4 [6337] 6337 innerTail822_4
    (if 3 ≤ innerCandidates822_3.length then search edge 3 [6337, 8101] innerCandidates822_3 14438 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan822_4

def innerTail822_2 : List Nat := 7933 :: innerTail822_3
def innerCandidates822_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates822_2_checked : candidates edge 4 6337 7933 innerTail822_3 best.bound = innerCandidates822_2 := by
  unfold candidates
  have row : edge 7933 = SearchData.row7933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan822_2 : scan edge (search edge 3) 4 [6337] 6337 innerTail822_2 best = best := by
  rw [innerTail822_2, scan, if_neg (by decide), innerCandidates822_2_checked]
  change scan edge (search edge 3) 4 [6337] 6337 innerTail822_3
    (if 3 ≤ innerCandidates822_2.length then search edge 3 [6337, 7933] innerCandidates822_2 14270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan822_3

def innerTail822_1 : List Nat := 6991 :: innerTail822_2
def innerCandidates822_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates822_1_checked : candidates edge 4 6337 6991 innerTail822_2 best.bound = innerCandidates822_1 := by
  unfold candidates
  have row : edge 6991 = SearchData.row6991.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan822_1 : scan edge (search edge 3) 4 [6337] 6337 innerTail822_1 best = best := by
  rw [innerTail822_1, scan, if_neg (by decide), innerCandidates822_1_checked]
  change scan edge (search edge 3) 4 [6337] 6337 innerTail822_2
    (if 3 ≤ innerCandidates822_1.length then search edge 3 [6337, 6991] innerCandidates822_1 13328 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan822_2

def innerTail822_0 : List Nat := 6949 :: innerTail822_1
def innerCandidates822_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates822_0_checked : candidates edge 4 6337 6949 innerTail822_1 best.bound = innerCandidates822_0 := by
  unfold candidates
  have row : edge 6949 = SearchData.row6949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan822_0 : scan edge (search edge 3) 4 [6337] 6337 innerTail822_0 best = best := by
  rw [innerTail822_0, scan, if_neg (by decide), innerCandidates822_0_checked]
  change scan edge (search edge 3) 4 [6337] 6337 innerTail822_1
    (if 3 ≤ innerCandidates822_0.length then search edge 3 [6337, 6949] innerCandidates822_0 13286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan822_1

theorem rootBranch822 : search edge 4 [6337] rootCandidates822 6337 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6337] 6337 innerTail822_0 best = best
  exact innerScan822_0

def rootCandidates821 : List Nat := [6551, 6983, 7331, 7589, 7907, 7937]
set_option maxRecDepth 4096 in
theorem rootCandidates821_checked : candidates edge 5 0 6329 tail822 best.bound = rootCandidates821 := by
  unfold candidates
  have row : edge 6329 = SearchData.row6329.contains := by funext q; rfl
  rw [row]
  decide

def innerTail821_6 : List Nat := []
theorem innerScan821_6 : scan edge (search edge 3) 4 [6329] 6329 innerTail821_6 best = best := rfl

def innerTail821_5 : List Nat := 7937 :: innerTail821_6
def innerCandidates821_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates821_5_checked : candidates edge 4 6329 7937 innerTail821_6 best.bound = innerCandidates821_5 := by
  unfold candidates
  have row : edge 7937 = SearchData.row7937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan821_5 : scan edge (search edge 3) 4 [6329] 6329 innerTail821_5 best = best := by
  rw [innerTail821_5, scan, if_neg (by decide), innerCandidates821_5_checked]
  change scan edge (search edge 3) 4 [6329] 6329 innerTail821_6
    (if 3 ≤ innerCandidates821_5.length then search edge 3 [6329, 7937] innerCandidates821_5 14266 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan821_6

def innerTail821_4 : List Nat := 7907 :: innerTail821_5
def innerCandidates821_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates821_4_checked : candidates edge 4 6329 7907 innerTail821_5 best.bound = innerCandidates821_4 := by
  unfold candidates
  have row : edge 7907 = SearchData.row7907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan821_4 : scan edge (search edge 3) 4 [6329] 6329 innerTail821_4 best = best := by
  rw [innerTail821_4, scan, if_neg (by decide), innerCandidates821_4_checked]
  change scan edge (search edge 3) 4 [6329] 6329 innerTail821_5
    (if 3 ≤ innerCandidates821_4.length then search edge 3 [6329, 7907] innerCandidates821_4 14236 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan821_5

def innerTail821_3 : List Nat := 7589 :: innerTail821_4
def innerCandidates821_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates821_3_checked : candidates edge 4 6329 7589 innerTail821_4 best.bound = innerCandidates821_3 := by
  unfold candidates
  have row : edge 7589 = SearchData.row7589.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan821_3 : scan edge (search edge 3) 4 [6329] 6329 innerTail821_3 best = best := by
  rw [innerTail821_3, scan, if_neg (by decide), innerCandidates821_3_checked]
  change scan edge (search edge 3) 4 [6329] 6329 innerTail821_4
    (if 3 ≤ innerCandidates821_3.length then search edge 3 [6329, 7589] innerCandidates821_3 13918 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan821_4

def innerTail821_2 : List Nat := 7331 :: innerTail821_3
def innerCandidates821_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates821_2_checked : candidates edge 4 6329 7331 innerTail821_3 best.bound = innerCandidates821_2 := by
  unfold candidates
  have row : edge 7331 = SearchData.row7331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan821_2 : scan edge (search edge 3) 4 [6329] 6329 innerTail821_2 best = best := by
  rw [innerTail821_2, scan, if_neg (by decide), innerCandidates821_2_checked]
  change scan edge (search edge 3) 4 [6329] 6329 innerTail821_3
    (if 3 ≤ innerCandidates821_2.length then search edge 3 [6329, 7331] innerCandidates821_2 13660 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan821_3

def innerTail821_1 : List Nat := 6983 :: innerTail821_2
def innerCandidates821_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates821_1_checked : candidates edge 4 6329 6983 innerTail821_2 best.bound = innerCandidates821_1 := by
  unfold candidates
  have row : edge 6983 = SearchData.row6983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan821_1 : scan edge (search edge 3) 4 [6329] 6329 innerTail821_1 best = best := by
  rw [innerTail821_1, scan, if_neg (by decide), innerCandidates821_1_checked]
  change scan edge (search edge 3) 4 [6329] 6329 innerTail821_2
    (if 3 ≤ innerCandidates821_1.length then search edge 3 [6329, 6983] innerCandidates821_1 13312 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan821_2

def innerTail821_0 : List Nat := 6551 :: innerTail821_1
def innerCandidates821_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates821_0_checked : candidates edge 4 6329 6551 innerTail821_1 best.bound = innerCandidates821_0 := by
  unfold candidates
  have row : edge 6551 = SearchData.row6551.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan821_0 : scan edge (search edge 3) 4 [6329] 6329 innerTail821_0 best = best := by
  rw [innerTail821_0, scan, if_neg (by decide), innerCandidates821_0_checked]
  change scan edge (search edge 3) 4 [6329] 6329 innerTail821_1
    (if 3 ≤ innerCandidates821_0.length then search edge 3 [6329, 6551] innerCandidates821_0 12880 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan821_1

theorem rootBranch821 : search edge 4 [6329] rootCandidates821 6329 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6329] 6329 innerTail821_0 best = best
  exact innerScan821_0

def rootCandidates820 : List Nat := [6491, 7019, 7523, 7919, 8237]
set_option maxRecDepth 4096 in
theorem rootCandidates820_checked : candidates edge 5 0 6323 tail821 best.bound = rootCandidates820 := by
  unfold candidates
  have row : edge 6323 = SearchData.row6323.contains := by funext q; rfl
  rw [row]
  decide

def innerTail820_5 : List Nat := []
theorem innerScan820_5 : scan edge (search edge 3) 4 [6323] 6323 innerTail820_5 best = best := rfl

def innerTail820_4 : List Nat := 8237 :: innerTail820_5
def innerCandidates820_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates820_4_checked : candidates edge 4 6323 8237 innerTail820_5 best.bound = innerCandidates820_4 := by
  unfold candidates
  have row : edge 8237 = SearchData.row8237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan820_4 : scan edge (search edge 3) 4 [6323] 6323 innerTail820_4 best = best := by
  rw [innerTail820_4, scan, if_neg (by decide), innerCandidates820_4_checked]
  change scan edge (search edge 3) 4 [6323] 6323 innerTail820_5
    (if 3 ≤ innerCandidates820_4.length then search edge 3 [6323, 8237] innerCandidates820_4 14560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan820_5

def innerTail820_3 : List Nat := 7919 :: innerTail820_4
def innerCandidates820_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates820_3_checked : candidates edge 4 6323 7919 innerTail820_4 best.bound = innerCandidates820_3 := by
  unfold candidates
  have row : edge 7919 = SearchData.row7919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan820_3 : scan edge (search edge 3) 4 [6323] 6323 innerTail820_3 best = best := by
  rw [innerTail820_3, scan, if_neg (by decide), innerCandidates820_3_checked]
  change scan edge (search edge 3) 4 [6323] 6323 innerTail820_4
    (if 3 ≤ innerCandidates820_3.length then search edge 3 [6323, 7919] innerCandidates820_3 14242 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan820_4

def innerTail820_2 : List Nat := 7523 :: innerTail820_3
def innerCandidates820_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates820_2_checked : candidates edge 4 6323 7523 innerTail820_3 best.bound = innerCandidates820_2 := by
  unfold candidates
  have row : edge 7523 = SearchData.row7523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan820_2 : scan edge (search edge 3) 4 [6323] 6323 innerTail820_2 best = best := by
  rw [innerTail820_2, scan, if_neg (by decide), innerCandidates820_2_checked]
  change scan edge (search edge 3) 4 [6323] 6323 innerTail820_3
    (if 3 ≤ innerCandidates820_2.length then search edge 3 [6323, 7523] innerCandidates820_2 13846 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan820_3

def innerTail820_1 : List Nat := 7019 :: innerTail820_2
def innerCandidates820_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates820_1_checked : candidates edge 4 6323 7019 innerTail820_2 best.bound = innerCandidates820_1 := by
  unfold candidates
  have row : edge 7019 = SearchData.row7019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan820_1 : scan edge (search edge 3) 4 [6323] 6323 innerTail820_1 best = best := by
  rw [innerTail820_1, scan, if_neg (by decide), innerCandidates820_1_checked]
  change scan edge (search edge 3) 4 [6323] 6323 innerTail820_2
    (if 3 ≤ innerCandidates820_1.length then search edge 3 [6323, 7019] innerCandidates820_1 13342 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan820_2

def innerTail820_0 : List Nat := 6491 :: innerTail820_1
def innerCandidates820_0 : List Nat := [8237]
set_option maxRecDepth 4096 in
theorem innerCandidates820_0_checked : candidates edge 4 6323 6491 innerTail820_1 best.bound = innerCandidates820_0 := by
  unfold candidates
  have row : edge 6491 = SearchData.row6491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan820_0 : scan edge (search edge 3) 4 [6323] 6323 innerTail820_0 best = best := by
  rw [innerTail820_0, scan, if_neg (by decide), innerCandidates820_0_checked]
  change scan edge (search edge 3) 4 [6323] 6323 innerTail820_1
    (if 3 ≤ innerCandidates820_0.length then search edge 3 [6323, 6491] innerCandidates820_0 12814 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan820_1

theorem rootBranch820 : search edge 4 [6323] rootCandidates820 6323 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6323] 6323 innerTail820_0 best = best
  exact innerScan820_0

def rootCandidates819 : List Nat := [6599, 6983, 7529, 7607, 7853, 8039]
set_option maxRecDepth 4096 in
theorem rootCandidates819_checked : candidates edge 5 0 6317 tail820 best.bound = rootCandidates819 := by
  unfold candidates
  have row : edge 6317 = SearchData.row6317.contains := by funext q; rfl
  rw [row]
  decide

def innerTail819_6 : List Nat := []
theorem innerScan819_6 : scan edge (search edge 3) 4 [6317] 6317 innerTail819_6 best = best := rfl

def innerTail819_5 : List Nat := 8039 :: innerTail819_6
def innerCandidates819_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates819_5_checked : candidates edge 4 6317 8039 innerTail819_6 best.bound = innerCandidates819_5 := by
  unfold candidates
  have row : edge 8039 = SearchData.row8039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan819_5 : scan edge (search edge 3) 4 [6317] 6317 innerTail819_5 best = best := by
  rw [innerTail819_5, scan, if_neg (by decide), innerCandidates819_5_checked]
  change scan edge (search edge 3) 4 [6317] 6317 innerTail819_6
    (if 3 ≤ innerCandidates819_5.length then search edge 3 [6317, 8039] innerCandidates819_5 14356 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan819_6

def innerTail819_4 : List Nat := 7853 :: innerTail819_5
def innerCandidates819_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates819_4_checked : candidates edge 4 6317 7853 innerTail819_5 best.bound = innerCandidates819_4 := by
  unfold candidates
  have row : edge 7853 = SearchData.row7853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan819_4 : scan edge (search edge 3) 4 [6317] 6317 innerTail819_4 best = best := by
  rw [innerTail819_4, scan, if_neg (by decide), innerCandidates819_4_checked]
  change scan edge (search edge 3) 4 [6317] 6317 innerTail819_5
    (if 3 ≤ innerCandidates819_4.length then search edge 3 [6317, 7853] innerCandidates819_4 14170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan819_5

def innerTail819_3 : List Nat := 7607 :: innerTail819_4
def innerCandidates819_3 : List Nat := [7853]
set_option maxRecDepth 4096 in
theorem innerCandidates819_3_checked : candidates edge 4 6317 7607 innerTail819_4 best.bound = innerCandidates819_3 := by
  unfold candidates
  have row : edge 7607 = SearchData.row7607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan819_3 : scan edge (search edge 3) 4 [6317] 6317 innerTail819_3 best = best := by
  rw [innerTail819_3, scan, if_neg (by decide), innerCandidates819_3_checked]
  change scan edge (search edge 3) 4 [6317] 6317 innerTail819_4
    (if 3 ≤ innerCandidates819_3.length then search edge 3 [6317, 7607] innerCandidates819_3 13924 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan819_4

def innerTail819_2 : List Nat := 7529 :: innerTail819_3
def innerCandidates819_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates819_2_checked : candidates edge 4 6317 7529 innerTail819_3 best.bound = innerCandidates819_2 := by
  unfold candidates
  have row : edge 7529 = SearchData.row7529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan819_2 : scan edge (search edge 3) 4 [6317] 6317 innerTail819_2 best = best := by
  rw [innerTail819_2, scan, if_neg (by decide), innerCandidates819_2_checked]
  change scan edge (search edge 3) 4 [6317] 6317 innerTail819_3
    (if 3 ≤ innerCandidates819_2.length then search edge 3 [6317, 7529] innerCandidates819_2 13846 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan819_3

def innerTail819_1 : List Nat := 6983 :: innerTail819_2
def innerCandidates819_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates819_1_checked : candidates edge 4 6317 6983 innerTail819_2 best.bound = innerCandidates819_1 := by
  unfold candidates
  have row : edge 6983 = SearchData.row6983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan819_1 : scan edge (search edge 3) 4 [6317] 6317 innerTail819_1 best = best := by
  rw [innerTail819_1, scan, if_neg (by decide), innerCandidates819_1_checked]
  change scan edge (search edge 3) 4 [6317] 6317 innerTail819_2
    (if 3 ≤ innerCandidates819_1.length then search edge 3 [6317, 6983] innerCandidates819_1 13300 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan819_2

def innerTail819_0 : List Nat := 6599 :: innerTail819_1
def innerCandidates819_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates819_0_checked : candidates edge 4 6317 6599 innerTail819_1 best.bound = innerCandidates819_0 := by
  unfold candidates
  have row : edge 6599 = SearchData.row6599.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan819_0 : scan edge (search edge 3) 4 [6317] 6317 innerTail819_0 best = best := by
  rw [innerTail819_0, scan, if_neg (by decide), innerCandidates819_0_checked]
  change scan edge (search edge 3) 4 [6317] 6317 innerTail819_1
    (if 3 ≤ innerCandidates819_0.length then search edge 3 [6317, 6599] innerCandidates819_0 12916 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan819_1

theorem rootBranch819 : search edge 4 [6317] rootCandidates819 6317 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6317] 6317 innerTail819_0 best = best
  exact innerScan819_0

def rootCandidates818 : List Nat := [6491, 6737, 6863, 6947]
set_option maxRecDepth 4096 in
theorem rootCandidates818_checked : candidates edge 5 0 6311 tail819 best.bound = rootCandidates818 := by
  unfold candidates
  have row : edge 6311 = SearchData.row6311.contains := by funext q; rfl
  rw [row]
  decide

def innerTail818_4 : List Nat := []
theorem innerScan818_4 : scan edge (search edge 3) 4 [6311] 6311 innerTail818_4 best = best := rfl

def innerTail818_3 : List Nat := 6947 :: innerTail818_4
def innerCandidates818_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates818_3_checked : candidates edge 4 6311 6947 innerTail818_4 best.bound = innerCandidates818_3 := by
  unfold candidates
  have row : edge 6947 = SearchData.row6947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan818_3 : scan edge (search edge 3) 4 [6311] 6311 innerTail818_3 best = best := by
  rw [innerTail818_3, scan, if_neg (by decide), innerCandidates818_3_checked]
  change scan edge (search edge 3) 4 [6311] 6311 innerTail818_4
    (if 3 ≤ innerCandidates818_3.length then search edge 3 [6311, 6947] innerCandidates818_3 13258 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan818_4

def innerTail818_2 : List Nat := 6863 :: innerTail818_3
def innerCandidates818_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates818_2_checked : candidates edge 4 6311 6863 innerTail818_3 best.bound = innerCandidates818_2 := by
  unfold candidates
  have row : edge 6863 = SearchData.row6863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan818_2 : scan edge (search edge 3) 4 [6311] 6311 innerTail818_2 best = best := by
  rw [innerTail818_2, scan, if_neg (by decide), innerCandidates818_2_checked]
  change scan edge (search edge 3) 4 [6311] 6311 innerTail818_3
    (if 3 ≤ innerCandidates818_2.length then search edge 3 [6311, 6863] innerCandidates818_2 13174 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan818_3

def innerTail818_1 : List Nat := 6737 :: innerTail818_2
def innerCandidates818_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates818_1_checked : candidates edge 4 6311 6737 innerTail818_2 best.bound = innerCandidates818_1 := by
  unfold candidates
  have row : edge 6737 = SearchData.row6737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan818_1 : scan edge (search edge 3) 4 [6311] 6311 innerTail818_1 best = best := by
  rw [innerTail818_1, scan, if_neg (by decide), innerCandidates818_1_checked]
  change scan edge (search edge 3) 4 [6311] 6311 innerTail818_2
    (if 3 ≤ innerCandidates818_1.length then search edge 3 [6311, 6737] innerCandidates818_1 13048 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan818_2

def innerTail818_0 : List Nat := 6491 :: innerTail818_1
def innerCandidates818_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates818_0_checked : candidates edge 4 6311 6491 innerTail818_1 best.bound = innerCandidates818_0 := by
  unfold candidates
  have row : edge 6491 = SearchData.row6491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan818_0 : scan edge (search edge 3) 4 [6311] 6311 innerTail818_0 best = best := by
  rw [innerTail818_0, scan, if_neg (by decide), innerCandidates818_0_checked]
  change scan edge (search edge 3) 4 [6311] 6311 innerTail818_1
    (if 3 ≤ innerCandidates818_0.length then search edge 3 [6311, 6491] innerCandidates818_0 12802 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan818_1

theorem rootBranch818 : search edge 4 [6311] rootCandidates818 6311 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6311] 6311 innerTail818_0 best = best
  exact innerScan818_0

def rootCandidates817 : List Nat := [6607, 7027, 7039, 7321, 7417, 7639, 8101, 8233]
set_option maxRecDepth 4096 in
theorem rootCandidates817_checked : candidates edge 5 0 6301 tail818 best.bound = rootCandidates817 := by
  unfold candidates
  have row : edge 6301 = SearchData.row6301.contains := by funext q; rfl
  rw [row]
  decide

def innerTail817_8 : List Nat := []
theorem innerScan817_8 : scan edge (search edge 3) 4 [6301] 6301 innerTail817_8 best = best := rfl

def innerTail817_7 : List Nat := 8233 :: innerTail817_8
def innerCandidates817_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates817_7_checked : candidates edge 4 6301 8233 innerTail817_8 best.bound = innerCandidates817_7 := by
  unfold candidates
  have row : edge 8233 = SearchData.row8233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan817_7 : scan edge (search edge 3) 4 [6301] 6301 innerTail817_7 best = best := by
  rw [innerTail817_7, scan, if_neg (by decide), innerCandidates817_7_checked]
  change scan edge (search edge 3) 4 [6301] 6301 innerTail817_8
    (if 3 ≤ innerCandidates817_7.length then search edge 3 [6301, 8233] innerCandidates817_7 14534 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan817_8

def innerTail817_6 : List Nat := 8101 :: innerTail817_7
def innerCandidates817_6 : List Nat := [8233]
set_option maxRecDepth 4096 in
theorem innerCandidates817_6_checked : candidates edge 4 6301 8101 innerTail817_7 best.bound = innerCandidates817_6 := by
  unfold candidates
  have row : edge 8101 = SearchData.row8101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan817_6 : scan edge (search edge 3) 4 [6301] 6301 innerTail817_6 best = best := by
  rw [innerTail817_6, scan, if_neg (by decide), innerCandidates817_6_checked]
  change scan edge (search edge 3) 4 [6301] 6301 innerTail817_7
    (if 3 ≤ innerCandidates817_6.length then search edge 3 [6301, 8101] innerCandidates817_6 14402 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan817_7

def innerTail817_5 : List Nat := 7639 :: innerTail817_6
def innerCandidates817_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates817_5_checked : candidates edge 4 6301 7639 innerTail817_6 best.bound = innerCandidates817_5 := by
  unfold candidates
  have row : edge 7639 = SearchData.row7639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan817_5 : scan edge (search edge 3) 4 [6301] 6301 innerTail817_5 best = best := by
  rw [innerTail817_5, scan, if_neg (by decide), innerCandidates817_5_checked]
  change scan edge (search edge 3) 4 [6301] 6301 innerTail817_6
    (if 3 ≤ innerCandidates817_5.length then search edge 3 [6301, 7639] innerCandidates817_5 13940 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan817_6

def innerTail817_4 : List Nat := 7417 :: innerTail817_5
def innerCandidates817_4 : List Nat := [7639]
set_option maxRecDepth 4096 in
theorem innerCandidates817_4_checked : candidates edge 4 6301 7417 innerTail817_5 best.bound = innerCandidates817_4 := by
  unfold candidates
  have row : edge 7417 = SearchData.row7417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan817_4 : scan edge (search edge 3) 4 [6301] 6301 innerTail817_4 best = best := by
  rw [innerTail817_4, scan, if_neg (by decide), innerCandidates817_4_checked]
  change scan edge (search edge 3) 4 [6301] 6301 innerTail817_5
    (if 3 ≤ innerCandidates817_4.length then search edge 3 [6301, 7417] innerCandidates817_4 13718 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan817_5

def innerTail817_3 : List Nat := 7321 :: innerTail817_4
def innerCandidates817_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates817_3_checked : candidates edge 4 6301 7321 innerTail817_4 best.bound = innerCandidates817_3 := by
  unfold candidates
  have row : edge 7321 = SearchData.row7321.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan817_3 : scan edge (search edge 3) 4 [6301] 6301 innerTail817_3 best = best := by
  rw [innerTail817_3, scan, if_neg (by decide), innerCandidates817_3_checked]
  change scan edge (search edge 3) 4 [6301] 6301 innerTail817_4
    (if 3 ≤ innerCandidates817_3.length then search edge 3 [6301, 7321] innerCandidates817_3 13622 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan817_4

def innerTail817_2 : List Nat := 7039 :: innerTail817_3
def innerCandidates817_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates817_2_checked : candidates edge 4 6301 7039 innerTail817_3 best.bound = innerCandidates817_2 := by
  unfold candidates
  have row : edge 7039 = SearchData.row7039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan817_2 : scan edge (search edge 3) 4 [6301] 6301 innerTail817_2 best = best := by
  rw [innerTail817_2, scan, if_neg (by decide), innerCandidates817_2_checked]
  change scan edge (search edge 3) 4 [6301] 6301 innerTail817_3
    (if 3 ≤ innerCandidates817_2.length then search edge 3 [6301, 7039] innerCandidates817_2 13340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan817_3

def innerTail817_1 : List Nat := 7027 :: innerTail817_2
def innerCandidates817_1 : List Nat := [8233]
set_option maxRecDepth 4096 in
theorem innerCandidates817_1_checked : candidates edge 4 6301 7027 innerTail817_2 best.bound = innerCandidates817_1 := by
  unfold candidates
  have row : edge 7027 = SearchData.row7027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan817_1 : scan edge (search edge 3) 4 [6301] 6301 innerTail817_1 best = best := by
  rw [innerTail817_1, scan, if_neg (by decide), innerCandidates817_1_checked]
  change scan edge (search edge 3) 4 [6301] 6301 innerTail817_2
    (if 3 ≤ innerCandidates817_1.length then search edge 3 [6301, 7027] innerCandidates817_1 13328 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan817_2

def innerTail817_0 : List Nat := 6607 :: innerTail817_1
def innerCandidates817_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates817_0_checked : candidates edge 4 6301 6607 innerTail817_1 best.bound = innerCandidates817_0 := by
  unfold candidates
  have row : edge 6607 = SearchData.row6607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan817_0 : scan edge (search edge 3) 4 [6301] 6301 innerTail817_0 best = best := by
  rw [innerTail817_0, scan, if_neg (by decide), innerCandidates817_0_checked]
  change scan edge (search edge 3) 4 [6301] 6301 innerTail817_1
    (if 3 ≤ innerCandidates817_0.length then search edge 3 [6301, 6607] innerCandidates817_0 12908 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan817_1

theorem rootBranch817 : search edge 4 [6301] rootCandidates817 6301 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6301] 6301 innerTail817_0 best = best
  exact innerScan817_0

def rootCandidates816 : List Nat := [7283, 7589, 8147]
set_option maxRecDepth 4096 in
theorem rootCandidates816_checked : candidates edge 5 0 6299 tail817 best.bound = rootCandidates816 := by
  unfold candidates
  have row : edge 6299 = SearchData.row6299.contains := by funext q; rfl
  rw [row]
  decide

end ProjectEulerSolutions.PrimePairUpper.Search
