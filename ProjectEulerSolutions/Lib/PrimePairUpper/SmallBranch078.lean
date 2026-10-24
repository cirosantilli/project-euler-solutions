import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates486 : List Nat := [3613, 4651, 5011, 5227, 5407, 5449, 5881, 6277, 6361, 7321, 7759, 7993, 8191, 8377]
set_option maxRecDepth 4096 in
theorem rootCandidates486_checked : candidates edge 5 0 3499 tail487 best.bound = rootCandidates486 := by
  unfold candidates
  have row : edge 3499 = SearchData.row3499.contains := by funext q; rfl
  rw [row]
  decide

def innerTail486_14 : List Nat := []
theorem innerScan486_14 : scan edge (search edge 3) 4 [3499] 3499 innerTail486_14 best = best := rfl

def innerTail486_13 : List Nat := 8377 :: innerTail486_14
def innerCandidates486_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates486_13_checked : candidates edge 4 3499 8377 innerTail486_14 best.bound = innerCandidates486_13 := by
  unfold candidates
  have row : edge 8377 = SearchData.row8377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan486_13 : scan edge (search edge 3) 4 [3499] 3499 innerTail486_13 best = best := by
  rw [innerTail486_13, scan, if_neg (by decide), innerCandidates486_13_checked]
  change scan edge (search edge 3) 4 [3499] 3499 innerTail486_14
    (if 3 ≤ innerCandidates486_13.length then search edge 3 [3499, 8377] innerCandidates486_13 11876 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan486_14

def innerTail486_12 : List Nat := 8191 :: innerTail486_13
def innerCandidates486_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates486_12_checked : candidates edge 4 3499 8191 innerTail486_13 best.bound = innerCandidates486_12 := by
  unfold candidates
  have row : edge 8191 = SearchData.row8191.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan486_12 : scan edge (search edge 3) 4 [3499] 3499 innerTail486_12 best = best := by
  rw [innerTail486_12, scan, if_neg (by decide), innerCandidates486_12_checked]
  change scan edge (search edge 3) 4 [3499] 3499 innerTail486_13
    (if 3 ≤ innerCandidates486_12.length then search edge 3 [3499, 8191] innerCandidates486_12 11690 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan486_13

def innerTail486_11 : List Nat := 7993 :: innerTail486_12
def innerCandidates486_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates486_11_checked : candidates edge 4 3499 7993 innerTail486_12 best.bound = innerCandidates486_11 := by
  unfold candidates
  have row : edge 7993 = SearchData.row7993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan486_11 : scan edge (search edge 3) 4 [3499] 3499 innerTail486_11 best = best := by
  rw [innerTail486_11, scan, if_neg (by decide), innerCandidates486_11_checked]
  change scan edge (search edge 3) 4 [3499] 3499 innerTail486_12
    (if 3 ≤ innerCandidates486_11.length then search edge 3 [3499, 7993] innerCandidates486_11 11492 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan486_12

def innerTail486_10 : List Nat := 7759 :: innerTail486_11
def innerCandidates486_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates486_10_checked : candidates edge 4 3499 7759 innerTail486_11 best.bound = innerCandidates486_10 := by
  unfold candidates
  have row : edge 7759 = SearchData.row7759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan486_10 : scan edge (search edge 3) 4 [3499] 3499 innerTail486_10 best = best := by
  rw [innerTail486_10, scan, if_neg (by decide), innerCandidates486_10_checked]
  change scan edge (search edge 3) 4 [3499] 3499 innerTail486_11
    (if 3 ≤ innerCandidates486_10.length then search edge 3 [3499, 7759] innerCandidates486_10 11258 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan486_11

def innerTail486_9 : List Nat := 7321 :: innerTail486_10
def innerCandidates486_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates486_9_checked : candidates edge 4 3499 7321 innerTail486_10 best.bound = innerCandidates486_9 := by
  unfold candidates
  have row : edge 7321 = SearchData.row7321.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan486_9 : scan edge (search edge 3) 4 [3499] 3499 innerTail486_9 best = best := by
  rw [innerTail486_9, scan, if_neg (by decide), innerCandidates486_9_checked]
  change scan edge (search edge 3) 4 [3499] 3499 innerTail486_10
    (if 3 ≤ innerCandidates486_9.length then search edge 3 [3499, 7321] innerCandidates486_9 10820 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan486_10

def innerTail486_8 : List Nat := 6361 :: innerTail486_9
def innerCandidates486_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates486_8_checked : candidates edge 4 3499 6361 innerTail486_9 best.bound = innerCandidates486_8 := by
  unfold candidates
  have row : edge 6361 = SearchData.row6361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan486_8 : scan edge (search edge 3) 4 [3499] 3499 innerTail486_8 best = best := by
  rw [innerTail486_8, scan, if_neg (by decide), innerCandidates486_8_checked]
  change scan edge (search edge 3) 4 [3499] 3499 innerTail486_9
    (if 3 ≤ innerCandidates486_8.length then search edge 3 [3499, 6361] innerCandidates486_8 9860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan486_9

def innerTail486_7 : List Nat := 6277 :: innerTail486_8
def innerCandidates486_7 : List Nat := [7321]
set_option maxRecDepth 4096 in
theorem innerCandidates486_7_checked : candidates edge 4 3499 6277 innerTail486_8 best.bound = innerCandidates486_7 := by
  unfold candidates
  have row : edge 6277 = SearchData.row6277.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan486_7 : scan edge (search edge 3) 4 [3499] 3499 innerTail486_7 best = best := by
  rw [innerTail486_7, scan, if_neg (by decide), innerCandidates486_7_checked]
  change scan edge (search edge 3) 4 [3499] 3499 innerTail486_8
    (if 3 ≤ innerCandidates486_7.length then search edge 3 [3499, 6277] innerCandidates486_7 9776 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan486_8

def innerTail486_6 : List Nat := 5881 :: innerTail486_7
def innerCandidates486_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates486_6_checked : candidates edge 4 3499 5881 innerTail486_7 best.bound = innerCandidates486_6 := by
  unfold candidates
  have row : edge 5881 = SearchData.row5881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan486_6 : scan edge (search edge 3) 4 [3499] 3499 innerTail486_6 best = best := by
  rw [innerTail486_6, scan, if_neg (by decide), innerCandidates486_6_checked]
  change scan edge (search edge 3) 4 [3499] 3499 innerTail486_7
    (if 3 ≤ innerCandidates486_6.length then search edge 3 [3499, 5881] innerCandidates486_6 9380 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan486_7

def innerTail486_5 : List Nat := 5449 :: innerTail486_6
def innerCandidates486_5 : List Nat := [7321]
set_option maxRecDepth 4096 in
theorem innerCandidates486_5_checked : candidates edge 4 3499 5449 innerTail486_6 best.bound = innerCandidates486_5 := by
  unfold candidates
  have row : edge 5449 = SearchData.row5449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan486_5 : scan edge (search edge 3) 4 [3499] 3499 innerTail486_5 best = best := by
  rw [innerTail486_5, scan, if_neg (by decide), innerCandidates486_5_checked]
  change scan edge (search edge 3) 4 [3499] 3499 innerTail486_6
    (if 3 ≤ innerCandidates486_5.length then search edge 3 [3499, 5449] innerCandidates486_5 8948 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan486_6

def innerTail486_4 : List Nat := 5407 :: innerTail486_5
def innerCandidates486_4 : List Nat := [5449]
set_option maxRecDepth 4096 in
theorem innerCandidates486_4_checked : candidates edge 4 3499 5407 innerTail486_5 best.bound = innerCandidates486_4 := by
  unfold candidates
  have row : edge 5407 = SearchData.row5407.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan486_4 : scan edge (search edge 3) 4 [3499] 3499 innerTail486_4 best = best := by
  rw [innerTail486_4, scan, if_neg (by decide), innerCandidates486_4_checked]
  change scan edge (search edge 3) 4 [3499] 3499 innerTail486_5
    (if 3 ≤ innerCandidates486_4.length then search edge 3 [3499, 5407] innerCandidates486_4 8906 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan486_5

def innerTail486_3 : List Nat := 5227 :: innerTail486_4
def innerCandidates486_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates486_3_checked : candidates edge 4 3499 5227 innerTail486_4 best.bound = innerCandidates486_3 := by
  unfold candidates
  have row : edge 5227 = SearchData.row5227.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan486_3 : scan edge (search edge 3) 4 [3499] 3499 innerTail486_3 best = best := by
  rw [innerTail486_3, scan, if_neg (by decide), innerCandidates486_3_checked]
  change scan edge (search edge 3) 4 [3499] 3499 innerTail486_4
    (if 3 ≤ innerCandidates486_3.length then search edge 3 [3499, 5227] innerCandidates486_3 8726 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan486_4

def innerTail486_2 : List Nat := 5011 :: innerTail486_3
def innerCandidates486_2 : List Nat := [8191]
set_option maxRecDepth 4096 in
theorem innerCandidates486_2_checked : candidates edge 4 3499 5011 innerTail486_3 best.bound = innerCandidates486_2 := by
  unfold candidates
  have row : edge 5011 = SearchData.row5011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan486_2 : scan edge (search edge 3) 4 [3499] 3499 innerTail486_2 best = best := by
  rw [innerTail486_2, scan, if_neg (by decide), innerCandidates486_2_checked]
  change scan edge (search edge 3) 4 [3499] 3499 innerTail486_3
    (if 3 ≤ innerCandidates486_2.length then search edge 3 [3499, 5011] innerCandidates486_2 8510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan486_3

def innerTail486_1 : List Nat := 4651 :: innerTail486_2
def innerCandidates486_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates486_1_checked : candidates edge 4 3499 4651 innerTail486_2 best.bound = innerCandidates486_1 := by
  unfold candidates
  have row : edge 4651 = SearchData.row4651.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan486_1 : scan edge (search edge 3) 4 [3499] 3499 innerTail486_1 best = best := by
  rw [innerTail486_1, scan, if_neg (by decide), innerCandidates486_1_checked]
  change scan edge (search edge 3) 4 [3499] 3499 innerTail486_2
    (if 3 ≤ innerCandidates486_1.length then search edge 3 [3499, 4651] innerCandidates486_1 8150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan486_2

def innerTail486_0 : List Nat := 3613 :: innerTail486_1
def innerCandidates486_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates486_0_checked : candidates edge 4 3499 3613 innerTail486_1 best.bound = innerCandidates486_0 := by
  unfold candidates
  have row : edge 3613 = SearchData.row3613.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan486_0 : scan edge (search edge 3) 4 [3499] 3499 innerTail486_0 best = best := by
  rw [innerTail486_0, scan, if_neg (by decide), innerCandidates486_0_checked]
  change scan edge (search edge 3) 4 [3499] 3499 innerTail486_1
    (if 3 ≤ innerCandidates486_0.length then search edge 3 [3499, 3613] innerCandidates486_0 7112 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan486_1

theorem rootBranch486 : search edge 4 [3499] rootCandidates486 3499 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3499] 3499 innerTail486_0 best = best
  exact innerScan486_0

def rootCandidates485 : List Nat := [3761, 3947, 4259, 4349, 4931, 5003, 5171, 5843, 6221, 6701, 7121, 7331, 8111, 8123]
set_option maxRecDepth 4096 in
theorem rootCandidates485_checked : candidates edge 5 0 3491 tail486 best.bound = rootCandidates485 := by
  unfold candidates
  have row : edge 3491 = SearchData.row3491.contains := by funext q; rfl
  rw [row]
  decide

def innerTail485_14 : List Nat := []
theorem innerScan485_14 : scan edge (search edge 3) 4 [3491] 3491 innerTail485_14 best = best := rfl

def innerTail485_13 : List Nat := 8123 :: innerTail485_14
def innerCandidates485_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates485_13_checked : candidates edge 4 3491 8123 innerTail485_14 best.bound = innerCandidates485_13 := by
  unfold candidates
  have row : edge 8123 = SearchData.row8123.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan485_13 : scan edge (search edge 3) 4 [3491] 3491 innerTail485_13 best = best := by
  rw [innerTail485_13, scan, if_neg (by decide), innerCandidates485_13_checked]
  change scan edge (search edge 3) 4 [3491] 3491 innerTail485_14
    (if 3 ≤ innerCandidates485_13.length then search edge 3 [3491, 8123] innerCandidates485_13 11614 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan485_14

def innerTail485_12 : List Nat := 8111 :: innerTail485_13
def innerCandidates485_12 : List Nat := [8123]
set_option maxRecDepth 4096 in
theorem innerCandidates485_12_checked : candidates edge 4 3491 8111 innerTail485_13 best.bound = innerCandidates485_12 := by
  unfold candidates
  have row : edge 8111 = SearchData.row8111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan485_12 : scan edge (search edge 3) 4 [3491] 3491 innerTail485_12 best = best := by
  rw [innerTail485_12, scan, if_neg (by decide), innerCandidates485_12_checked]
  change scan edge (search edge 3) 4 [3491] 3491 innerTail485_13
    (if 3 ≤ innerCandidates485_12.length then search edge 3 [3491, 8111] innerCandidates485_12 11602 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan485_13

def innerTail485_11 : List Nat := 7331 :: innerTail485_12
def innerCandidates485_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates485_11_checked : candidates edge 4 3491 7331 innerTail485_12 best.bound = innerCandidates485_11 := by
  unfold candidates
  have row : edge 7331 = SearchData.row7331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan485_11 : scan edge (search edge 3) 4 [3491] 3491 innerTail485_11 best = best := by
  rw [innerTail485_11, scan, if_neg (by decide), innerCandidates485_11_checked]
  change scan edge (search edge 3) 4 [3491] 3491 innerTail485_12
    (if 3 ≤ innerCandidates485_11.length then search edge 3 [3491, 7331] innerCandidates485_11 10822 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan485_12

def innerTail485_10 : List Nat := 7121 :: innerTail485_11
def innerCandidates485_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates485_10_checked : candidates edge 4 3491 7121 innerTail485_11 best.bound = innerCandidates485_10 := by
  unfold candidates
  have row : edge 7121 = SearchData.row7121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan485_10 : scan edge (search edge 3) 4 [3491] 3491 innerTail485_10 best = best := by
  rw [innerTail485_10, scan, if_neg (by decide), innerCandidates485_10_checked]
  change scan edge (search edge 3) 4 [3491] 3491 innerTail485_11
    (if 3 ≤ innerCandidates485_10.length then search edge 3 [3491, 7121] innerCandidates485_10 10612 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan485_11

def innerTail485_9 : List Nat := 6701 :: innerTail485_10
def innerCandidates485_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates485_9_checked : candidates edge 4 3491 6701 innerTail485_10 best.bound = innerCandidates485_9 := by
  unfold candidates
  have row : edge 6701 = SearchData.row6701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan485_9 : scan edge (search edge 3) 4 [3491] 3491 innerTail485_9 best = best := by
  rw [innerTail485_9, scan, if_neg (by decide), innerCandidates485_9_checked]
  change scan edge (search edge 3) 4 [3491] 3491 innerTail485_10
    (if 3 ≤ innerCandidates485_9.length then search edge 3 [3491, 6701] innerCandidates485_9 10192 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan485_10

def innerTail485_8 : List Nat := 6221 :: innerTail485_9
def innerCandidates485_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates485_8_checked : candidates edge 4 3491 6221 innerTail485_9 best.bound = innerCandidates485_8 := by
  unfold candidates
  have row : edge 6221 = SearchData.row6221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan485_8 : scan edge (search edge 3) 4 [3491] 3491 innerTail485_8 best = best := by
  rw [innerTail485_8, scan, if_neg (by decide), innerCandidates485_8_checked]
  change scan edge (search edge 3) 4 [3491] 3491 innerTail485_9
    (if 3 ≤ innerCandidates485_8.length then search edge 3 [3491, 6221] innerCandidates485_8 9712 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan485_9

def innerTail485_7 : List Nat := 5843 :: innerTail485_8
def innerCandidates485_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates485_7_checked : candidates edge 4 3491 5843 innerTail485_8 best.bound = innerCandidates485_7 := by
  unfold candidates
  have row : edge 5843 = SearchData.row5843.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan485_7 : scan edge (search edge 3) 4 [3491] 3491 innerTail485_7 best = best := by
  rw [innerTail485_7, scan, if_neg (by decide), innerCandidates485_7_checked]
  change scan edge (search edge 3) 4 [3491] 3491 innerTail485_8
    (if 3 ≤ innerCandidates485_7.length then search edge 3 [3491, 5843] innerCandidates485_7 9334 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan485_8

def innerTail485_6 : List Nat := 5171 :: innerTail485_7
def innerCandidates485_6 : List Nat := [8111]
set_option maxRecDepth 4096 in
theorem innerCandidates485_6_checked : candidates edge 4 3491 5171 innerTail485_7 best.bound = innerCandidates485_6 := by
  unfold candidates
  have row : edge 5171 = SearchData.row5171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan485_6 : scan edge (search edge 3) 4 [3491] 3491 innerTail485_6 best = best := by
  rw [innerTail485_6, scan, if_neg (by decide), innerCandidates485_6_checked]
  change scan edge (search edge 3) 4 [3491] 3491 innerTail485_7
    (if 3 ≤ innerCandidates485_6.length then search edge 3 [3491, 5171] innerCandidates485_6 8662 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan485_7

def innerTail485_5 : List Nat := 5003 :: innerTail485_6
def innerCandidates485_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates485_5_checked : candidates edge 4 3491 5003 innerTail485_6 best.bound = innerCandidates485_5 := by
  unfold candidates
  have row : edge 5003 = SearchData.row5003.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan485_5 : scan edge (search edge 3) 4 [3491] 3491 innerTail485_5 best = best := by
  rw [innerTail485_5, scan, if_neg (by decide), innerCandidates485_5_checked]
  change scan edge (search edge 3) 4 [3491] 3491 innerTail485_6
    (if 3 ≤ innerCandidates485_5.length then search edge 3 [3491, 5003] innerCandidates485_5 8494 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan485_6

def innerTail485_4 : List Nat := 4931 :: innerTail485_5
def innerCandidates485_4 : List Nat := [5843]
set_option maxRecDepth 4096 in
theorem innerCandidates485_4_checked : candidates edge 4 3491 4931 innerTail485_5 best.bound = innerCandidates485_4 := by
  unfold candidates
  have row : edge 4931 = SearchData.row4931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan485_4 : scan edge (search edge 3) 4 [3491] 3491 innerTail485_4 best = best := by
  rw [innerTail485_4, scan, if_neg (by decide), innerCandidates485_4_checked]
  change scan edge (search edge 3) 4 [3491] 3491 innerTail485_5
    (if 3 ≤ innerCandidates485_4.length then search edge 3 [3491, 4931] innerCandidates485_4 8422 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan485_5

def innerTail485_3 : List Nat := 4349 :: innerTail485_4
def innerCandidates485_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates485_3_checked : candidates edge 4 3491 4349 innerTail485_4 best.bound = innerCandidates485_3 := by
  unfold candidates
  have row : edge 4349 = SearchData.row4349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan485_3 : scan edge (search edge 3) 4 [3491] 3491 innerTail485_3 best = best := by
  rw [innerTail485_3, scan, if_neg (by decide), innerCandidates485_3_checked]
  change scan edge (search edge 3) 4 [3491] 3491 innerTail485_4
    (if 3 ≤ innerCandidates485_3.length then search edge 3 [3491, 4349] innerCandidates485_3 7840 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan485_4

def innerTail485_2 : List Nat := 4259 :: innerTail485_3
def innerCandidates485_2 : List Nat := [6221]
set_option maxRecDepth 4096 in
theorem innerCandidates485_2_checked : candidates edge 4 3491 4259 innerTail485_3 best.bound = innerCandidates485_2 := by
  unfold candidates
  have row : edge 4259 = SearchData.row4259.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan485_2 : scan edge (search edge 3) 4 [3491] 3491 innerTail485_2 best = best := by
  rw [innerTail485_2, scan, if_neg (by decide), innerCandidates485_2_checked]
  change scan edge (search edge 3) 4 [3491] 3491 innerTail485_3
    (if 3 ≤ innerCandidates485_2.length then search edge 3 [3491, 4259] innerCandidates485_2 7750 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan485_3

def innerTail485_1 : List Nat := 3947 :: innerTail485_2
def innerCandidates485_1 : List Nat := [5171]
set_option maxRecDepth 4096 in
theorem innerCandidates485_1_checked : candidates edge 4 3491 3947 innerTail485_2 best.bound = innerCandidates485_1 := by
  unfold candidates
  have row : edge 3947 = SearchData.row3947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan485_1 : scan edge (search edge 3) 4 [3491] 3491 innerTail485_1 best = best := by
  rw [innerTail485_1, scan, if_neg (by decide), innerCandidates485_1_checked]
  change scan edge (search edge 3) 4 [3491] 3491 innerTail485_2
    (if 3 ≤ innerCandidates485_1.length then search edge 3 [3491, 3947] innerCandidates485_1 7438 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan485_2

def innerTail485_0 : List Nat := 3761 :: innerTail485_1
def innerCandidates485_0 : List Nat := [5843]
set_option maxRecDepth 4096 in
theorem innerCandidates485_0_checked : candidates edge 4 3491 3761 innerTail485_1 best.bound = innerCandidates485_0 := by
  unfold candidates
  have row : edge 3761 = SearchData.row3761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan485_0 : scan edge (search edge 3) 4 [3491] 3491 innerTail485_0 best = best := by
  rw [innerTail485_0, scan, if_neg (by decide), innerCandidates485_0_checked]
  change scan edge (search edge 3) 4 [3491] 3491 innerTail485_1
    (if 3 ≤ innerCandidates485_0.length then search edge 3 [3491, 3761] innerCandidates485_0 7252 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan485_1

theorem rootBranch485 : search edge 4 [3491] rootCandidates485 3491 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3491] 3491 innerTail485_0 best = best
  exact innerScan485_0

def rootCandidates484 : List Nat := [3517, 4021, 4357, 4903, 5503, 5839, 6703, 7219, 7309, 7687, 8311]
set_option maxRecDepth 4096 in
theorem rootCandidates484_checked : candidates edge 5 0 3469 tail485 best.bound = rootCandidates484 := by
  unfold candidates
  have row : edge 3469 = SearchData.row3469.contains := by funext q; rfl
  rw [row]
  decide

def innerTail484_11 : List Nat := []
theorem innerScan484_11 : scan edge (search edge 3) 4 [3469] 3469 innerTail484_11 best = best := rfl

def innerTail484_10 : List Nat := 8311 :: innerTail484_11
def innerCandidates484_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates484_10_checked : candidates edge 4 3469 8311 innerTail484_11 best.bound = innerCandidates484_10 := by
  unfold candidates
  have row : edge 8311 = SearchData.row8311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan484_10 : scan edge (search edge 3) 4 [3469] 3469 innerTail484_10 best = best := by
  rw [innerTail484_10, scan, if_neg (by decide), innerCandidates484_10_checked]
  change scan edge (search edge 3) 4 [3469] 3469 innerTail484_11
    (if 3 ≤ innerCandidates484_10.length then search edge 3 [3469, 8311] innerCandidates484_10 11780 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan484_11

def innerTail484_9 : List Nat := 7687 :: innerTail484_10
def innerCandidates484_9 : List Nat := [8311]
set_option maxRecDepth 4096 in
theorem innerCandidates484_9_checked : candidates edge 4 3469 7687 innerTail484_10 best.bound = innerCandidates484_9 := by
  unfold candidates
  have row : edge 7687 = SearchData.row7687.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan484_9 : scan edge (search edge 3) 4 [3469] 3469 innerTail484_9 best = best := by
  rw [innerTail484_9, scan, if_neg (by decide), innerCandidates484_9_checked]
  change scan edge (search edge 3) 4 [3469] 3469 innerTail484_10
    (if 3 ≤ innerCandidates484_9.length then search edge 3 [3469, 7687] innerCandidates484_9 11156 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan484_10

def innerTail484_8 : List Nat := 7309 :: innerTail484_9
def innerCandidates484_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates484_8_checked : candidates edge 4 3469 7309 innerTail484_9 best.bound = innerCandidates484_8 := by
  unfold candidates
  have row : edge 7309 = SearchData.row7309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan484_8 : scan edge (search edge 3) 4 [3469] 3469 innerTail484_8 best = best := by
  rw [innerTail484_8, scan, if_neg (by decide), innerCandidates484_8_checked]
  change scan edge (search edge 3) 4 [3469] 3469 innerTail484_9
    (if 3 ≤ innerCandidates484_8.length then search edge 3 [3469, 7309] innerCandidates484_8 10778 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan484_9

def innerTail484_7 : List Nat := 7219 :: innerTail484_8
def innerCandidates484_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates484_7_checked : candidates edge 4 3469 7219 innerTail484_8 best.bound = innerCandidates484_7 := by
  unfold candidates
  have row : edge 7219 = SearchData.row7219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan484_7 : scan edge (search edge 3) 4 [3469] 3469 innerTail484_7 best = best := by
  rw [innerTail484_7, scan, if_neg (by decide), innerCandidates484_7_checked]
  change scan edge (search edge 3) 4 [3469] 3469 innerTail484_8
    (if 3 ≤ innerCandidates484_7.length then search edge 3 [3469, 7219] innerCandidates484_7 10688 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan484_8

def innerTail484_6 : List Nat := 6703 :: innerTail484_7
def innerCandidates484_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates484_6_checked : candidates edge 4 3469 6703 innerTail484_7 best.bound = innerCandidates484_6 := by
  unfold candidates
  have row : edge 6703 = SearchData.row6703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan484_6 : scan edge (search edge 3) 4 [3469] 3469 innerTail484_6 best = best := by
  rw [innerTail484_6, scan, if_neg (by decide), innerCandidates484_6_checked]
  change scan edge (search edge 3) 4 [3469] 3469 innerTail484_7
    (if 3 ≤ innerCandidates484_6.length then search edge 3 [3469, 6703] innerCandidates484_6 10172 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan484_7

def innerTail484_5 : List Nat := 5839 :: innerTail484_6
def innerCandidates484_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates484_5_checked : candidates edge 4 3469 5839 innerTail484_6 best.bound = innerCandidates484_5 := by
  unfold candidates
  have row : edge 5839 = SearchData.row5839.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan484_5 : scan edge (search edge 3) 4 [3469] 3469 innerTail484_5 best = best := by
  rw [innerTail484_5, scan, if_neg (by decide), innerCandidates484_5_checked]
  change scan edge (search edge 3) 4 [3469] 3469 innerTail484_6
    (if 3 ≤ innerCandidates484_5.length then search edge 3 [3469, 5839] innerCandidates484_5 9308 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan484_6

def innerTail484_4 : List Nat := 5503 :: innerTail484_5
def innerCandidates484_4 : List Nat := [8311]
set_option maxRecDepth 4096 in
theorem innerCandidates484_4_checked : candidates edge 4 3469 5503 innerTail484_5 best.bound = innerCandidates484_4 := by
  unfold candidates
  have row : edge 5503 = SearchData.row5503.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan484_4 : scan edge (search edge 3) 4 [3469] 3469 innerTail484_4 best = best := by
  rw [innerTail484_4, scan, if_neg (by decide), innerCandidates484_4_checked]
  change scan edge (search edge 3) 4 [3469] 3469 innerTail484_5
    (if 3 ≤ innerCandidates484_4.length then search edge 3 [3469, 5503] innerCandidates484_4 8972 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan484_5

def innerTail484_3 : List Nat := 4903 :: innerTail484_4
def innerCandidates484_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates484_3_checked : candidates edge 4 3469 4903 innerTail484_4 best.bound = innerCandidates484_3 := by
  unfold candidates
  have row : edge 4903 = SearchData.row4903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan484_3 : scan edge (search edge 3) 4 [3469] 3469 innerTail484_3 best = best := by
  rw [innerTail484_3, scan, if_neg (by decide), innerCandidates484_3_checked]
  change scan edge (search edge 3) 4 [3469] 3469 innerTail484_4
    (if 3 ≤ innerCandidates484_3.length then search edge 3 [3469, 4903] innerCandidates484_3 8372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan484_4

def innerTail484_2 : List Nat := 4357 :: innerTail484_3
def innerCandidates484_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates484_2_checked : candidates edge 4 3469 4357 innerTail484_3 best.bound = innerCandidates484_2 := by
  unfold candidates
  have row : edge 4357 = SearchData.row4357.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan484_2 : scan edge (search edge 3) 4 [3469] 3469 innerTail484_2 best = best := by
  rw [innerTail484_2, scan, if_neg (by decide), innerCandidates484_2_checked]
  change scan edge (search edge 3) 4 [3469] 3469 innerTail484_3
    (if 3 ≤ innerCandidates484_2.length then search edge 3 [3469, 4357] innerCandidates484_2 7826 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan484_3

def innerTail484_1 : List Nat := 4021 :: innerTail484_2
def innerCandidates484_1 : List Nat := [4903]
set_option maxRecDepth 4096 in
theorem innerCandidates484_1_checked : candidates edge 4 3469 4021 innerTail484_2 best.bound = innerCandidates484_1 := by
  unfold candidates
  have row : edge 4021 = SearchData.row4021.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan484_1 : scan edge (search edge 3) 4 [3469] 3469 innerTail484_1 best = best := by
  rw [innerTail484_1, scan, if_neg (by decide), innerCandidates484_1_checked]
  change scan edge (search edge 3) 4 [3469] 3469 innerTail484_2
    (if 3 ≤ innerCandidates484_1.length then search edge 3 [3469, 4021] innerCandidates484_1 7490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan484_2

def innerTail484_0 : List Nat := 3517 :: innerTail484_1
def innerCandidates484_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates484_0_checked : candidates edge 4 3469 3517 innerTail484_1 best.bound = innerCandidates484_0 := by
  unfold candidates
  have row : edge 3517 = SearchData.row3517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan484_0 : scan edge (search edge 3) 4 [3469] 3469 innerTail484_0 best = best := by
  rw [innerTail484_0, scan, if_neg (by decide), innerCandidates484_0_checked]
  change scan edge (search edge 3) 4 [3469] 3469 innerTail484_1
    (if 3 ≤ innerCandidates484_0.length then search edge 3 [3469, 3517] innerCandidates484_0 6986 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan484_1

theorem rootBranch484 : search edge 4 [3469] rootCandidates484 3469 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3469] 3469 innerTail484_0 best = best
  exact innerScan484_0

def rootCandidates483 : List Nat := [3623, 3851, 4457, 4799, 4817, 4919, 5009, 5231, 5273, 5297, 5471, 5591, 5807, 6113, 6599, 6959, 7001, 7481, 7949, 8117]
set_option maxRecDepth 4096 in
theorem rootCandidates483_checked : candidates edge 5 0 3467 tail484 best.bound = rootCandidates483 := by
  unfold candidates
  have row : edge 3467 = SearchData.row3467.contains := by funext q; rfl
  rw [row]
  decide

def innerTail483_20 : List Nat := []
theorem innerScan483_20 : scan edge (search edge 3) 4 [3467] 3467 innerTail483_20 best = best := rfl

def innerTail483_19 : List Nat := 8117 :: innerTail483_20
def innerCandidates483_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates483_19_checked : candidates edge 4 3467 8117 innerTail483_20 best.bound = innerCandidates483_19 := by
  unfold candidates
  have row : edge 8117 = SearchData.row8117.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan483_19 : scan edge (search edge 3) 4 [3467] 3467 innerTail483_19 best = best := by
  rw [innerTail483_19, scan, if_neg (by decide), innerCandidates483_19_checked]
  change scan edge (search edge 3) 4 [3467] 3467 innerTail483_20
    (if 3 ≤ innerCandidates483_19.length then search edge 3 [3467, 8117] innerCandidates483_19 11584 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan483_20

def innerTail483_18 : List Nat := 7949 :: innerTail483_19
def innerCandidates483_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates483_18_checked : candidates edge 4 3467 7949 innerTail483_19 best.bound = innerCandidates483_18 := by
  unfold candidates
  have row : edge 7949 = SearchData.row7949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan483_18 : scan edge (search edge 3) 4 [3467] 3467 innerTail483_18 best = best := by
  rw [innerTail483_18, scan, if_neg (by decide), innerCandidates483_18_checked]
  change scan edge (search edge 3) 4 [3467] 3467 innerTail483_19
    (if 3 ≤ innerCandidates483_18.length then search edge 3 [3467, 7949] innerCandidates483_18 11416 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan483_19

def innerTail483_17 : List Nat := 7481 :: innerTail483_18
def innerCandidates483_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates483_17_checked : candidates edge 4 3467 7481 innerTail483_18 best.bound = innerCandidates483_17 := by
  unfold candidates
  have row : edge 7481 = SearchData.row7481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan483_17 : scan edge (search edge 3) 4 [3467] 3467 innerTail483_17 best = best := by
  rw [innerTail483_17, scan, if_neg (by decide), innerCandidates483_17_checked]
  change scan edge (search edge 3) 4 [3467] 3467 innerTail483_18
    (if 3 ≤ innerCandidates483_17.length then search edge 3 [3467, 7481] innerCandidates483_17 10948 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan483_18

def innerTail483_16 : List Nat := 7001 :: innerTail483_17
def innerCandidates483_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates483_16_checked : candidates edge 4 3467 7001 innerTail483_17 best.bound = innerCandidates483_16 := by
  unfold candidates
  have row : edge 7001 = SearchData.row7001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan483_16 : scan edge (search edge 3) 4 [3467] 3467 innerTail483_16 best = best := by
  rw [innerTail483_16, scan, if_neg (by decide), innerCandidates483_16_checked]
  change scan edge (search edge 3) 4 [3467] 3467 innerTail483_17
    (if 3 ≤ innerCandidates483_16.length then search edge 3 [3467, 7001] innerCandidates483_16 10468 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan483_17

def innerTail483_15 : List Nat := 6959 :: innerTail483_16
def innerCandidates483_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates483_15_checked : candidates edge 4 3467 6959 innerTail483_16 best.bound = innerCandidates483_15 := by
  unfold candidates
  have row : edge 6959 = SearchData.row6959.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan483_15 : scan edge (search edge 3) 4 [3467] 3467 innerTail483_15 best = best := by
  rw [innerTail483_15, scan, if_neg (by decide), innerCandidates483_15_checked]
  change scan edge (search edge 3) 4 [3467] 3467 innerTail483_16
    (if 3 ≤ innerCandidates483_15.length then search edge 3 [3467, 6959] innerCandidates483_15 10426 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan483_16

def innerTail483_14 : List Nat := 6599 :: innerTail483_15
def innerCandidates483_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates483_14_checked : candidates edge 4 3467 6599 innerTail483_15 best.bound = innerCandidates483_14 := by
  unfold candidates
  have row : edge 6599 = SearchData.row6599.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan483_14 : scan edge (search edge 3) 4 [3467] 3467 innerTail483_14 best = best := by
  rw [innerTail483_14, scan, if_neg (by decide), innerCandidates483_14_checked]
  change scan edge (search edge 3) 4 [3467] 3467 innerTail483_15
    (if 3 ≤ innerCandidates483_14.length then search edge 3 [3467, 6599] innerCandidates483_14 10066 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan483_15

def innerTail483_13 : List Nat := 6113 :: innerTail483_14
def innerCandidates483_13 : List Nat := [7001]
set_option maxRecDepth 4096 in
theorem innerCandidates483_13_checked : candidates edge 4 3467 6113 innerTail483_14 best.bound = innerCandidates483_13 := by
  unfold candidates
  have row : edge 6113 = SearchData.row6113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan483_13 : scan edge (search edge 3) 4 [3467] 3467 innerTail483_13 best = best := by
  rw [innerTail483_13, scan, if_neg (by decide), innerCandidates483_13_checked]
  change scan edge (search edge 3) 4 [3467] 3467 innerTail483_14
    (if 3 ≤ innerCandidates483_13.length then search edge 3 [3467, 6113] innerCandidates483_13 9580 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan483_14

def innerTail483_12 : List Nat := 5807 :: innerTail483_13
def innerCandidates483_12 : List Nat := [7949]
set_option maxRecDepth 4096 in
theorem innerCandidates483_12_checked : candidates edge 4 3467 5807 innerTail483_13 best.bound = innerCandidates483_12 := by
  unfold candidates
  have row : edge 5807 = SearchData.row5807.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan483_12 : scan edge (search edge 3) 4 [3467] 3467 innerTail483_12 best = best := by
  rw [innerTail483_12, scan, if_neg (by decide), innerCandidates483_12_checked]
  change scan edge (search edge 3) 4 [3467] 3467 innerTail483_13
    (if 3 ≤ innerCandidates483_12.length then search edge 3 [3467, 5807] innerCandidates483_12 9274 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan483_13

def innerTail483_11 : List Nat := 5591 :: innerTail483_12
def innerCandidates483_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates483_11_checked : candidates edge 4 3467 5591 innerTail483_12 best.bound = innerCandidates483_11 := by
  unfold candidates
  have row : edge 5591 = SearchData.row5591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan483_11 : scan edge (search edge 3) 4 [3467] 3467 innerTail483_11 best = best := by
  rw [innerTail483_11, scan, if_neg (by decide), innerCandidates483_11_checked]
  change scan edge (search edge 3) 4 [3467] 3467 innerTail483_12
    (if 3 ≤ innerCandidates483_11.length then search edge 3 [3467, 5591] innerCandidates483_11 9058 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan483_12

def innerTail483_10 : List Nat := 5471 :: innerTail483_11
def innerCandidates483_10 : List Nat := [5807]
set_option maxRecDepth 4096 in
theorem innerCandidates483_10_checked : candidates edge 4 3467 5471 innerTail483_11 best.bound = innerCandidates483_10 := by
  unfold candidates
  have row : edge 5471 = SearchData.row5471.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan483_10 : scan edge (search edge 3) 4 [3467] 3467 innerTail483_10 best = best := by
  rw [innerTail483_10, scan, if_neg (by decide), innerCandidates483_10_checked]
  change scan edge (search edge 3) 4 [3467] 3467 innerTail483_11
    (if 3 ≤ innerCandidates483_10.length then search edge 3 [3467, 5471] innerCandidates483_10 8938 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan483_11

def innerTail483_9 : List Nat := 5297 :: innerTail483_10
def innerCandidates483_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates483_9_checked : candidates edge 4 3467 5297 innerTail483_10 best.bound = innerCandidates483_9 := by
  unfold candidates
  have row : edge 5297 = SearchData.row5297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan483_9 : scan edge (search edge 3) 4 [3467] 3467 innerTail483_9 best = best := by
  rw [innerTail483_9, scan, if_neg (by decide), innerCandidates483_9_checked]
  change scan edge (search edge 3) 4 [3467] 3467 innerTail483_10
    (if 3 ≤ innerCandidates483_9.length then search edge 3 [3467, 5297] innerCandidates483_9 8764 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan483_10

def innerTail483_8 : List Nat := 5273 :: innerTail483_9
def innerCandidates483_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates483_8_checked : candidates edge 4 3467 5273 innerTail483_9 best.bound = innerCandidates483_8 := by
  unfold candidates
  have row : edge 5273 = SearchData.row5273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan483_8 : scan edge (search edge 3) 4 [3467] 3467 innerTail483_8 best = best := by
  rw [innerTail483_8, scan, if_neg (by decide), innerCandidates483_8_checked]
  change scan edge (search edge 3) 4 [3467] 3467 innerTail483_9
    (if 3 ≤ innerCandidates483_8.length then search edge 3 [3467, 5273] innerCandidates483_8 8740 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan483_9

def innerTail483_7 : List Nat := 5231 :: innerTail483_8
def innerCandidates483_7 : List Nat := [7949]
set_option maxRecDepth 4096 in
theorem innerCandidates483_7_checked : candidates edge 4 3467 5231 innerTail483_8 best.bound = innerCandidates483_7 := by
  unfold candidates
  have row : edge 5231 = SearchData.row5231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan483_7 : scan edge (search edge 3) 4 [3467] 3467 innerTail483_7 best = best := by
  rw [innerTail483_7, scan, if_neg (by decide), innerCandidates483_7_checked]
  change scan edge (search edge 3) 4 [3467] 3467 innerTail483_8
    (if 3 ≤ innerCandidates483_7.length then search edge 3 [3467, 5231] innerCandidates483_7 8698 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan483_8

def innerTail483_6 : List Nat := 5009 :: innerTail483_7
def innerCandidates483_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates483_6_checked : candidates edge 4 3467 5009 innerTail483_7 best.bound = innerCandidates483_6 := by
  unfold candidates
  have row : edge 5009 = SearchData.row5009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan483_6 : scan edge (search edge 3) 4 [3467] 3467 innerTail483_6 best = best := by
  rw [innerTail483_6, scan, if_neg (by decide), innerCandidates483_6_checked]
  change scan edge (search edge 3) 4 [3467] 3467 innerTail483_7
    (if 3 ≤ innerCandidates483_6.length then search edge 3 [3467, 5009] innerCandidates483_6 8476 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan483_7

def innerTail483_5 : List Nat := 4919 :: innerTail483_6
def innerCandidates483_5 : List Nat := [6599, 7481]
set_option maxRecDepth 4096 in
theorem innerCandidates483_5_checked : candidates edge 4 3467 4919 innerTail483_6 best.bound = innerCandidates483_5 := by
  unfold candidates
  have row : edge 4919 = SearchData.row4919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan483_5 : scan edge (search edge 3) 4 [3467] 3467 innerTail483_5 best = best := by
  rw [innerTail483_5, scan, if_neg (by decide), innerCandidates483_5_checked]
  change scan edge (search edge 3) 4 [3467] 3467 innerTail483_6
    (if 3 ≤ innerCandidates483_5.length then search edge 3 [3467, 4919] innerCandidates483_5 8386 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan483_6

def innerTail483_4 : List Nat := 4817 :: innerTail483_5
def innerCandidates483_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates483_4_checked : candidates edge 4 3467 4817 innerTail483_5 best.bound = innerCandidates483_4 := by
  unfold candidates
  have row : edge 4817 = SearchData.row4817.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan483_4 : scan edge (search edge 3) 4 [3467] 3467 innerTail483_4 best = best := by
  rw [innerTail483_4, scan, if_neg (by decide), innerCandidates483_4_checked]
  change scan edge (search edge 3) 4 [3467] 3467 innerTail483_5
    (if 3 ≤ innerCandidates483_4.length then search edge 3 [3467, 4817] innerCandidates483_4 8284 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan483_5

def innerTail483_3 : List Nat := 4799 :: innerTail483_4
def innerCandidates483_3 : List Nat := [5231]
set_option maxRecDepth 4096 in
theorem innerCandidates483_3_checked : candidates edge 4 3467 4799 innerTail483_4 best.bound = innerCandidates483_3 := by
  unfold candidates
  have row : edge 4799 = SearchData.row4799.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan483_3 : scan edge (search edge 3) 4 [3467] 3467 innerTail483_3 best = best := by
  rw [innerTail483_3, scan, if_neg (by decide), innerCandidates483_3_checked]
  change scan edge (search edge 3) 4 [3467] 3467 innerTail483_4
    (if 3 ≤ innerCandidates483_3.length then search edge 3 [3467, 4799] innerCandidates483_3 8266 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan483_4

def innerTail483_2 : List Nat := 4457 :: innerTail483_3
def innerCandidates483_2 : List Nat := [5591]
set_option maxRecDepth 4096 in
theorem innerCandidates483_2_checked : candidates edge 4 3467 4457 innerTail483_3 best.bound = innerCandidates483_2 := by
  unfold candidates
  have row : edge 4457 = SearchData.row4457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan483_2 : scan edge (search edge 3) 4 [3467] 3467 innerTail483_2 best = best := by
  rw [innerTail483_2, scan, if_neg (by decide), innerCandidates483_2_checked]
  change scan edge (search edge 3) 4 [3467] 3467 innerTail483_3
    (if 3 ≤ innerCandidates483_2.length then search edge 3 [3467, 4457] innerCandidates483_2 7924 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan483_3

def innerTail483_1 : List Nat := 3851 :: innerTail483_2
def innerCandidates483_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates483_1_checked : candidates edge 4 3467 3851 innerTail483_2 best.bound = innerCandidates483_1 := by
  unfold candidates
  have row : edge 3851 = SearchData.row3851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan483_1 : scan edge (search edge 3) 4 [3467] 3467 innerTail483_1 best = best := by
  rw [innerTail483_1, scan, if_neg (by decide), innerCandidates483_1_checked]
  change scan edge (search edge 3) 4 [3467] 3467 innerTail483_2
    (if 3 ≤ innerCandidates483_1.length then search edge 3 [3467, 3851] innerCandidates483_1 7318 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan483_2

def innerTail483_0 : List Nat := 3623 :: innerTail483_1
def innerCandidates483_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates483_0_checked : candidates edge 4 3467 3623 innerTail483_1 best.bound = innerCandidates483_0 := by
  unfold candidates
  have row : edge 3623 = SearchData.row3623.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan483_0 : scan edge (search edge 3) 4 [3467] 3467 innerTail483_0 best = best := by
  rw [innerTail483_0, scan, if_neg (by decide), innerCandidates483_0_checked]
  change scan edge (search edge 3) 4 [3467] 3467 innerTail483_1
    (if 3 ≤ innerCandidates483_0.length then search edge 3 [3467, 3623] innerCandidates483_0 7090 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan483_1

theorem rootBranch483 : search edge 4 [3467] rootCandidates483 3467 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3467] 3467 innerTail483_0 best = best
  exact innerScan483_0

end ProjectEulerSolutions.PrimePairUpper.Search
