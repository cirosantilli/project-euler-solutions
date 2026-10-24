import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates159 : List Nat := [1367, 1523, 2027, 2333, 2609, 2687, 3533, 4337, 4799, 4871, 5039, 5417, 5651, 6029, 7109, 7457, 7727]
set_option maxRecDepth 4096 in
theorem rootCandidates159_checked : candidates edge 5 0 953 tail160 best.bound = rootCandidates159 := by
  unfold candidates
  have row : edge 953 = PrimePairCache.smallRow953.contains := by funext q; rfl
  rw [row]
  decide

def innerTail159_17 : List Nat := []
theorem innerScan159_17 : scan edge (search edge 3) 4 [953] 953 innerTail159_17 best = best := rfl

def innerTail159_16 : List Nat := 7727 :: innerTail159_17
def innerCandidates159_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates159_16_checked : candidates edge 4 953 7727 innerTail159_17 best.bound = innerCandidates159_16 := by
  unfold candidates
  have row : edge 7727 = SearchData.row7727.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan159_16 : scan edge (search edge 3) 4 [953] 953 innerTail159_16 best = best := by
  rw [innerTail159_16, scan, if_neg (by decide), innerCandidates159_16_checked]
  change scan edge (search edge 3) 4 [953] 953 innerTail159_17
    (if 3 ≤ innerCandidates159_16.length then search edge 3 [953, 7727] innerCandidates159_16 8680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan159_17

def innerTail159_15 : List Nat := 7457 :: innerTail159_16
def innerCandidates159_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates159_15_checked : candidates edge 4 953 7457 innerTail159_16 best.bound = innerCandidates159_15 := by
  unfold candidates
  have row : edge 7457 = SearchData.row7457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan159_15 : scan edge (search edge 3) 4 [953] 953 innerTail159_15 best = best := by
  rw [innerTail159_15, scan, if_neg (by decide), innerCandidates159_15_checked]
  change scan edge (search edge 3) 4 [953] 953 innerTail159_16
    (if 3 ≤ innerCandidates159_15.length then search edge 3 [953, 7457] innerCandidates159_15 8410 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan159_16

def innerTail159_14 : List Nat := 7109 :: innerTail159_15
def innerCandidates159_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates159_14_checked : candidates edge 4 953 7109 innerTail159_15 best.bound = innerCandidates159_14 := by
  unfold candidates
  have row : edge 7109 = SearchData.row7109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan159_14 : scan edge (search edge 3) 4 [953] 953 innerTail159_14 best = best := by
  rw [innerTail159_14, scan, if_neg (by decide), innerCandidates159_14_checked]
  change scan edge (search edge 3) 4 [953] 953 innerTail159_15
    (if 3 ≤ innerCandidates159_14.length then search edge 3 [953, 7109] innerCandidates159_14 8062 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan159_15

def innerTail159_13 : List Nat := 6029 :: innerTail159_14
def innerCandidates159_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates159_13_checked : candidates edge 4 953 6029 innerTail159_14 best.bound = innerCandidates159_13 := by
  unfold candidates
  have row : edge 6029 = SearchData.row6029.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan159_13 : scan edge (search edge 3) 4 [953] 953 innerTail159_13 best = best := by
  rw [innerTail159_13, scan, if_neg (by decide), innerCandidates159_13_checked]
  change scan edge (search edge 3) 4 [953] 953 innerTail159_14
    (if 3 ≤ innerCandidates159_13.length then search edge 3 [953, 6029] innerCandidates159_13 6982 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan159_14

def innerTail159_12 : List Nat := 5651 :: innerTail159_13
def innerCandidates159_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates159_12_checked : candidates edge 4 953 5651 innerTail159_13 best.bound = innerCandidates159_12 := by
  unfold candidates
  have row : edge 5651 = SearchData.row5651.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan159_12 : scan edge (search edge 3) 4 [953] 953 innerTail159_12 best = best := by
  rw [innerTail159_12, scan, if_neg (by decide), innerCandidates159_12_checked]
  change scan edge (search edge 3) 4 [953] 953 innerTail159_13
    (if 3 ≤ innerCandidates159_12.length then search edge 3 [953, 5651] innerCandidates159_12 6604 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan159_13

def innerTail159_11 : List Nat := 5417 :: innerTail159_12
def innerCandidates159_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates159_11_checked : candidates edge 4 953 5417 innerTail159_12 best.bound = innerCandidates159_11 := by
  unfold candidates
  have row : edge 5417 = SearchData.row5417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan159_11 : scan edge (search edge 3) 4 [953] 953 innerTail159_11 best = best := by
  rw [innerTail159_11, scan, if_neg (by decide), innerCandidates159_11_checked]
  change scan edge (search edge 3) 4 [953] 953 innerTail159_12
    (if 3 ≤ innerCandidates159_11.length then search edge 3 [953, 5417] innerCandidates159_11 6370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan159_12

def innerTail159_10 : List Nat := 5039 :: innerTail159_11
def innerCandidates159_10 : List Nat := [7727]
set_option maxRecDepth 4096 in
theorem innerCandidates159_10_checked : candidates edge 4 953 5039 innerTail159_11 best.bound = innerCandidates159_10 := by
  unfold candidates
  have row : edge 5039 = SearchData.row5039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan159_10 : scan edge (search edge 3) 4 [953] 953 innerTail159_10 best = best := by
  rw [innerTail159_10, scan, if_neg (by decide), innerCandidates159_10_checked]
  change scan edge (search edge 3) 4 [953] 953 innerTail159_11
    (if 3 ≤ innerCandidates159_10.length then search edge 3 [953, 5039] innerCandidates159_10 5992 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan159_11

def innerTail159_9 : List Nat := 4871 :: innerTail159_10
def innerCandidates159_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates159_9_checked : candidates edge 4 953 4871 innerTail159_10 best.bound = innerCandidates159_9 := by
  unfold candidates
  have row : edge 4871 = SearchData.row4871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan159_9 : scan edge (search edge 3) 4 [953] 953 innerTail159_9 best = best := by
  rw [innerTail159_9, scan, if_neg (by decide), innerCandidates159_9_checked]
  change scan edge (search edge 3) 4 [953] 953 innerTail159_10
    (if 3 ≤ innerCandidates159_9.length then search edge 3 [953, 4871] innerCandidates159_9 5824 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan159_10

def innerTail159_8 : List Nat := 4799 :: innerTail159_9
def innerCandidates159_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates159_8_checked : candidates edge 4 953 4799 innerTail159_9 best.bound = innerCandidates159_8 := by
  unfold candidates
  have row : edge 4799 = SearchData.row4799.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan159_8 : scan edge (search edge 3) 4 [953] 953 innerTail159_8 best = best := by
  rw [innerTail159_8, scan, if_neg (by decide), innerCandidates159_8_checked]
  change scan edge (search edge 3) 4 [953] 953 innerTail159_9
    (if 3 ≤ innerCandidates159_8.length then search edge 3 [953, 4799] innerCandidates159_8 5752 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan159_9

def innerTail159_7 : List Nat := 4337 :: innerTail159_8
def innerCandidates159_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates159_7_checked : candidates edge 4 953 4337 innerTail159_8 best.bound = innerCandidates159_7 := by
  unfold candidates
  have row : edge 4337 = SearchData.row4337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan159_7 : scan edge (search edge 3) 4 [953] 953 innerTail159_7 best = best := by
  rw [innerTail159_7, scan, if_neg (by decide), innerCandidates159_7_checked]
  change scan edge (search edge 3) 4 [953] 953 innerTail159_8
    (if 3 ≤ innerCandidates159_7.length then search edge 3 [953, 4337] innerCandidates159_7 5290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan159_8

def innerTail159_6 : List Nat := 3533 :: innerTail159_7
def innerCandidates159_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates159_6_checked : candidates edge 4 953 3533 innerTail159_7 best.bound = innerCandidates159_6 := by
  unfold candidates
  have row : edge 3533 = SearchData.row3533.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan159_6 : scan edge (search edge 3) 4 [953] 953 innerTail159_6 best = best := by
  rw [innerTail159_6, scan, if_neg (by decide), innerCandidates159_6_checked]
  change scan edge (search edge 3) 4 [953] 953 innerTail159_7
    (if 3 ≤ innerCandidates159_6.length then search edge 3 [953, 3533] innerCandidates159_6 4486 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan159_7

def innerTail159_5 : List Nat := 2687 :: innerTail159_6
def innerCandidates159_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates159_5_checked : candidates edge 4 953 2687 innerTail159_6 best.bound = innerCandidates159_5 := by
  unfold candidates
  have row : edge 2687 = SearchData.row2687.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan159_5 : scan edge (search edge 3) 4 [953] 953 innerTail159_5 best = best := by
  rw [innerTail159_5, scan, if_neg (by decide), innerCandidates159_5_checked]
  change scan edge (search edge 3) 4 [953] 953 innerTail159_6
    (if 3 ≤ innerCandidates159_5.length then search edge 3 [953, 2687] innerCandidates159_5 3640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan159_6

def innerTail159_4 : List Nat := 2609 :: innerTail159_5
def innerCandidates159_4 : List Nat := [5651]
set_option maxRecDepth 4096 in
theorem innerCandidates159_4_checked : candidates edge 4 953 2609 innerTail159_5 best.bound = innerCandidates159_4 := by
  unfold candidates
  have row : edge 2609 = PrimePairCache.smallRow2609.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan159_4 : scan edge (search edge 3) 4 [953] 953 innerTail159_4 best = best := by
  rw [innerTail159_4, scan, if_neg (by decide), innerCandidates159_4_checked]
  change scan edge (search edge 3) 4 [953] 953 innerTail159_5
    (if 3 ≤ innerCandidates159_4.length then search edge 3 [953, 2609] innerCandidates159_4 3562 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan159_5

def innerTail159_3 : List Nat := 2333 :: innerTail159_4
def innerCandidates159_3 : List Nat := [3533]
set_option maxRecDepth 4096 in
theorem innerCandidates159_3_checked : candidates edge 4 953 2333 innerTail159_4 best.bound = innerCandidates159_3 := by
  unfold candidates
  have row : edge 2333 = PrimePairCache.smallRow2333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan159_3 : scan edge (search edge 3) 4 [953] 953 innerTail159_3 best = best := by
  rw [innerTail159_3, scan, if_neg (by decide), innerCandidates159_3_checked]
  change scan edge (search edge 3) 4 [953] 953 innerTail159_4
    (if 3 ≤ innerCandidates159_3.length then search edge 3 [953, 2333] innerCandidates159_3 3286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan159_4

def innerTail159_2 : List Nat := 2027 :: innerTail159_3
def innerCandidates159_2 : List Nat := [7457]
set_option maxRecDepth 4096 in
theorem innerCandidates159_2_checked : candidates edge 4 953 2027 innerTail159_3 best.bound = innerCandidates159_2 := by
  unfold candidates
  have row : edge 2027 = PrimePairCache.smallRow2027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan159_2 : scan edge (search edge 3) 4 [953] 953 innerTail159_2 best = best := by
  rw [innerTail159_2, scan, if_neg (by decide), innerCandidates159_2_checked]
  change scan edge (search edge 3) 4 [953] 953 innerTail159_3
    (if 3 ≤ innerCandidates159_2.length then search edge 3 [953, 2027] innerCandidates159_2 2980 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan159_3

def innerTail159_1 : List Nat := 1523 :: innerTail159_2
def innerCandidates159_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates159_1_checked : candidates edge 4 953 1523 innerTail159_2 best.bound = innerCandidates159_1 := by
  unfold candidates
  have row : edge 1523 = PrimePairCache.smallRow1523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan159_1 : scan edge (search edge 3) 4 [953] 953 innerTail159_1 best = best := by
  rw [innerTail159_1, scan, if_neg (by decide), innerCandidates159_1_checked]
  change scan edge (search edge 3) 4 [953] 953 innerTail159_2
    (if 3 ≤ innerCandidates159_1.length then search edge 3 [953, 1523] innerCandidates159_1 2476 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan159_2

def innerTail159_0 : List Nat := 1367 :: innerTail159_1
def innerCandidates159_0 : List Nat := [2027]
set_option maxRecDepth 4096 in
theorem innerCandidates159_0_checked : candidates edge 4 953 1367 innerTail159_1 best.bound = innerCandidates159_0 := by
  unfold candidates
  have row : edge 1367 = PrimePairCache.smallRow1367.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan159_0 : scan edge (search edge 3) 4 [953] 953 innerTail159_0 best = best := by
  rw [innerTail159_0, scan, if_neg (by decide), innerCandidates159_0_checked]
  change scan edge (search edge 3) 4 [953] 953 innerTail159_1
    (if 3 ≤ innerCandidates159_0.length then search edge 3 [953, 1367] innerCandidates159_0 2320 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan159_1

theorem rootBranch159 : search edge 4 [953] rootCandidates159 953 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [953] 953 innerTail159_0 best = best
  exact innerScan159_0

def rootCandidates158 : List Nat := [1097, 1439, 1901, 1949, 2657, 3041, 3491, 3623, 4523, 4637, 4931, 5009, 5051, 5981, 6947, 8111, 8387]
set_option maxRecDepth 4096 in
theorem rootCandidates158_checked : candidates edge 5 0 947 tail159 best.bound = rootCandidates158 := by
  unfold candidates
  have row : edge 947 = PrimePairCache.smallRow947.contains := by funext q; rfl
  rw [row]
  decide

def innerTail158_17 : List Nat := []
theorem innerScan158_17 : scan edge (search edge 3) 4 [947] 947 innerTail158_17 best = best := rfl

def innerTail158_16 : List Nat := 8387 :: innerTail158_17
def innerCandidates158_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates158_16_checked : candidates edge 4 947 8387 innerTail158_17 best.bound = innerCandidates158_16 := by
  unfold candidates
  have row : edge 8387 = SearchData.row8387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan158_16 : scan edge (search edge 3) 4 [947] 947 innerTail158_16 best = best := by
  rw [innerTail158_16, scan, if_neg (by decide), innerCandidates158_16_checked]
  change scan edge (search edge 3) 4 [947] 947 innerTail158_17
    (if 3 ≤ innerCandidates158_16.length then search edge 3 [947, 8387] innerCandidates158_16 9334 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan158_17

def innerTail158_15 : List Nat := 8111 :: innerTail158_16
def innerCandidates158_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates158_15_checked : candidates edge 4 947 8111 innerTail158_16 best.bound = innerCandidates158_15 := by
  unfold candidates
  have row : edge 8111 = SearchData.row8111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan158_15 : scan edge (search edge 3) 4 [947] 947 innerTail158_15 best = best := by
  rw [innerTail158_15, scan, if_neg (by decide), innerCandidates158_15_checked]
  change scan edge (search edge 3) 4 [947] 947 innerTail158_16
    (if 3 ≤ innerCandidates158_15.length then search edge 3 [947, 8111] innerCandidates158_15 9058 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan158_16

def innerTail158_14 : List Nat := 6947 :: innerTail158_15
def innerCandidates158_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates158_14_checked : candidates edge 4 947 6947 innerTail158_15 best.bound = innerCandidates158_14 := by
  unfold candidates
  have row : edge 6947 = SearchData.row6947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan158_14 : scan edge (search edge 3) 4 [947] 947 innerTail158_14 best = best := by
  rw [innerTail158_14, scan, if_neg (by decide), innerCandidates158_14_checked]
  change scan edge (search edge 3) 4 [947] 947 innerTail158_15
    (if 3 ≤ innerCandidates158_14.length then search edge 3 [947, 6947] innerCandidates158_14 7894 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan158_15

def innerTail158_13 : List Nat := 5981 :: innerTail158_14
def innerCandidates158_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates158_13_checked : candidates edge 4 947 5981 innerTail158_14 best.bound = innerCandidates158_13 := by
  unfold candidates
  have row : edge 5981 = SearchData.row5981.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan158_13 : scan edge (search edge 3) 4 [947] 947 innerTail158_13 best = best := by
  rw [innerTail158_13, scan, if_neg (by decide), innerCandidates158_13_checked]
  change scan edge (search edge 3) 4 [947] 947 innerTail158_14
    (if 3 ≤ innerCandidates158_13.length then search edge 3 [947, 5981] innerCandidates158_13 6928 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan158_14

def innerTail158_12 : List Nat := 5051 :: innerTail158_13
def innerCandidates158_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates158_12_checked : candidates edge 4 947 5051 innerTail158_13 best.bound = innerCandidates158_12 := by
  unfold candidates
  have row : edge 5051 = SearchData.row5051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan158_12 : scan edge (search edge 3) 4 [947] 947 innerTail158_12 best = best := by
  rw [innerTail158_12, scan, if_neg (by decide), innerCandidates158_12_checked]
  change scan edge (search edge 3) 4 [947] 947 innerTail158_13
    (if 3 ≤ innerCandidates158_12.length then search edge 3 [947, 5051] innerCandidates158_12 5998 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan158_13

def innerTail158_11 : List Nat := 5009 :: innerTail158_12
def innerCandidates158_11 : List Nat := [6947]
set_option maxRecDepth 4096 in
theorem innerCandidates158_11_checked : candidates edge 4 947 5009 innerTail158_12 best.bound = innerCandidates158_11 := by
  unfold candidates
  have row : edge 5009 = SearchData.row5009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan158_11 : scan edge (search edge 3) 4 [947] 947 innerTail158_11 best = best := by
  rw [innerTail158_11, scan, if_neg (by decide), innerCandidates158_11_checked]
  change scan edge (search edge 3) 4 [947] 947 innerTail158_12
    (if 3 ≤ innerCandidates158_11.length then search edge 3 [947, 5009] innerCandidates158_11 5956 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan158_12

def innerTail158_10 : List Nat := 4931 :: innerTail158_11
def innerCandidates158_10 : List Nat := [5009]
set_option maxRecDepth 4096 in
theorem innerCandidates158_10_checked : candidates edge 4 947 4931 innerTail158_11 best.bound = innerCandidates158_10 := by
  unfold candidates
  have row : edge 4931 = SearchData.row4931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan158_10 : scan edge (search edge 3) 4 [947] 947 innerTail158_10 best = best := by
  rw [innerTail158_10, scan, if_neg (by decide), innerCandidates158_10_checked]
  change scan edge (search edge 3) 4 [947] 947 innerTail158_11
    (if 3 ≤ innerCandidates158_10.length then search edge 3 [947, 4931] innerCandidates158_10 5878 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan158_11

def innerTail158_9 : List Nat := 4637 :: innerTail158_10
def innerCandidates158_9 : List Nat := [5009]
set_option maxRecDepth 4096 in
theorem innerCandidates158_9_checked : candidates edge 4 947 4637 innerTail158_10 best.bound = innerCandidates158_9 := by
  unfold candidates
  have row : edge 4637 = SearchData.row4637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan158_9 : scan edge (search edge 3) 4 [947] 947 innerTail158_9 best = best := by
  rw [innerTail158_9, scan, if_neg (by decide), innerCandidates158_9_checked]
  change scan edge (search edge 3) 4 [947] 947 innerTail158_10
    (if 3 ≤ innerCandidates158_9.length then search edge 3 [947, 4637] innerCandidates158_9 5584 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan158_10

def innerTail158_8 : List Nat := 4523 :: innerTail158_9
def innerCandidates158_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates158_8_checked : candidates edge 4 947 4523 innerTail158_9 best.bound = innerCandidates158_8 := by
  unfold candidates
  have row : edge 4523 = SearchData.row4523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan158_8 : scan edge (search edge 3) 4 [947] 947 innerTail158_8 best = best := by
  rw [innerTail158_8, scan, if_neg (by decide), innerCandidates158_8_checked]
  change scan edge (search edge 3) 4 [947] 947 innerTail158_9
    (if 3 ≤ innerCandidates158_8.length then search edge 3 [947, 4523] innerCandidates158_8 5470 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan158_9

def innerTail158_7 : List Nat := 3623 :: innerTail158_8
def innerCandidates158_7 : List Nat := [5981]
set_option maxRecDepth 4096 in
theorem innerCandidates158_7_checked : candidates edge 4 947 3623 innerTail158_8 best.bound = innerCandidates158_7 := by
  unfold candidates
  have row : edge 3623 = SearchData.row3623.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan158_7 : scan edge (search edge 3) 4 [947] 947 innerTail158_7 best = best := by
  rw [innerTail158_7, scan, if_neg (by decide), innerCandidates158_7_checked]
  change scan edge (search edge 3) 4 [947] 947 innerTail158_8
    (if 3 ≤ innerCandidates158_7.length then search edge 3 [947, 3623] innerCandidates158_7 4570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan158_8

def innerTail158_6 : List Nat := 3491 :: innerTail158_7
def innerCandidates158_6 : List Nat := [4931, 8111]
set_option maxRecDepth 4096 in
theorem innerCandidates158_6_checked : candidates edge 4 947 3491 innerTail158_7 best.bound = innerCandidates158_6 := by
  unfold candidates
  have row : edge 3491 = SearchData.row3491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan158_6 : scan edge (search edge 3) 4 [947] 947 innerTail158_6 best = best := by
  rw [innerTail158_6, scan, if_neg (by decide), innerCandidates158_6_checked]
  change scan edge (search edge 3) 4 [947] 947 innerTail158_7
    (if 3 ≤ innerCandidates158_6.length then search edge 3 [947, 3491] innerCandidates158_6 4438 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan158_7

def innerTail158_5 : List Nat := 3041 :: innerTail158_6
def innerCandidates158_5 : List Nat := [3491]
set_option maxRecDepth 4096 in
theorem innerCandidates158_5_checked : candidates edge 4 947 3041 innerTail158_6 best.bound = innerCandidates158_5 := by
  unfold candidates
  have row : edge 3041 = SearchData.row3041.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan158_5 : scan edge (search edge 3) 4 [947] 947 innerTail158_5 best = best := by
  rw [innerTail158_5, scan, if_neg (by decide), innerCandidates158_5_checked]
  change scan edge (search edge 3) 4 [947] 947 innerTail158_6
    (if 3 ≤ innerCandidates158_5.length then search edge 3 [947, 3041] innerCandidates158_5 3988 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan158_6

def innerTail158_4 : List Nat := 2657 :: innerTail158_5
def innerCandidates158_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates158_4_checked : candidates edge 4 947 2657 innerTail158_5 best.bound = innerCandidates158_4 := by
  unfold candidates
  have row : edge 2657 = PrimePairCache.smallRow2657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan158_4 : scan edge (search edge 3) 4 [947] 947 innerTail158_4 best = best := by
  rw [innerTail158_4, scan, if_neg (by decide), innerCandidates158_4_checked]
  change scan edge (search edge 3) 4 [947] 947 innerTail158_5
    (if 3 ≤ innerCandidates158_4.length then search edge 3 [947, 2657] innerCandidates158_4 3604 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan158_5

def innerTail158_3 : List Nat := 1949 :: innerTail158_4
def innerCandidates158_3 : List Nat := [2657]
set_option maxRecDepth 4096 in
theorem innerCandidates158_3_checked : candidates edge 4 947 1949 innerTail158_4 best.bound = innerCandidates158_3 := by
  unfold candidates
  have row : edge 1949 = PrimePairCache.smallRow1949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan158_3 : scan edge (search edge 3) 4 [947] 947 innerTail158_3 best = best := by
  rw [innerTail158_3, scan, if_neg (by decide), innerCandidates158_3_checked]
  change scan edge (search edge 3) 4 [947] 947 innerTail158_4
    (if 3 ≤ innerCandidates158_3.length then search edge 3 [947, 1949] innerCandidates158_3 2896 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan158_4

def innerTail158_2 : List Nat := 1901 :: innerTail158_3
def innerCandidates158_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates158_2_checked : candidates edge 4 947 1901 innerTail158_3 best.bound = innerCandidates158_2 := by
  unfold candidates
  have row : edge 1901 = PrimePairCache.smallRow1901.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan158_2 : scan edge (search edge 3) 4 [947] 947 innerTail158_2 best = best := by
  rw [innerTail158_2, scan, if_neg (by decide), innerCandidates158_2_checked]
  change scan edge (search edge 3) 4 [947] 947 innerTail158_3
    (if 3 ≤ innerCandidates158_2.length then search edge 3 [947, 1901] innerCandidates158_2 2848 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan158_3

def innerTail158_1 : List Nat := 1439 :: innerTail158_2
def innerCandidates158_1 : List Nat := [4931]
set_option maxRecDepth 4096 in
theorem innerCandidates158_1_checked : candidates edge 4 947 1439 innerTail158_2 best.bound = innerCandidates158_1 := by
  unfold candidates
  have row : edge 1439 = PrimePairCache.smallRow1439.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan158_1 : scan edge (search edge 3) 4 [947] 947 innerTail158_1 best = best := by
  rw [innerTail158_1, scan, if_neg (by decide), innerCandidates158_1_checked]
  change scan edge (search edge 3) 4 [947] 947 innerTail158_2
    (if 3 ≤ innerCandidates158_1.length then search edge 3 [947, 1439] innerCandidates158_1 2386 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan158_2

def innerTail158_0 : List Nat := 1097 :: innerTail158_1
def innerCandidates158_0 : List Nat := [4637]
set_option maxRecDepth 4096 in
theorem innerCandidates158_0_checked : candidates edge 4 947 1097 innerTail158_1 best.bound = innerCandidates158_0 := by
  unfold candidates
  have row : edge 1097 = PrimePairCache.smallRow1097.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan158_0 : scan edge (search edge 3) 4 [947] 947 innerTail158_0 best = best := by
  rw [innerTail158_0, scan, if_neg (by decide), innerCandidates158_0_checked]
  change scan edge (search edge 3) 4 [947] 947 innerTail158_1
    (if 3 ≤ innerCandidates158_0.length then search edge 3 [947, 1097] innerCandidates158_0 2044 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan158_1

theorem rootBranch158 : search edge 4 [947] rootCandidates158 947 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [947] 947 innerTail158_0 best = best
  exact innerScan158_0

def rootCandidates157 : List Nat := [1049, 1187, 1229, 1511, 1607, 1709, 2099, 3137, 3221, 3389, 3527, 4409, 4649, 4817, 6047, 6053, 6263, 6299, 6971, 7229, 8369]
set_option maxRecDepth 4096 in
theorem rootCandidates157_checked : candidates edge 5 0 941 tail158 best.bound = rootCandidates157 := by
  unfold candidates
  have row : edge 941 = PrimePairCache.smallRow941.contains := by funext q; rfl
  rw [row]
  decide

def innerTail157_21 : List Nat := []
theorem innerScan157_21 : scan edge (search edge 3) 4 [941] 941 innerTail157_21 best = best := rfl

def innerTail157_20 : List Nat := 8369 :: innerTail157_21
def innerCandidates157_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates157_20_checked : candidates edge 4 941 8369 innerTail157_21 best.bound = innerCandidates157_20 := by
  unfold candidates
  have row : edge 8369 = SearchData.row8369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan157_20 : scan edge (search edge 3) 4 [941] 941 innerTail157_20 best = best := by
  rw [innerTail157_20, scan, if_neg (by decide), innerCandidates157_20_checked]
  change scan edge (search edge 3) 4 [941] 941 innerTail157_21
    (if 3 ≤ innerCandidates157_20.length then search edge 3 [941, 8369] innerCandidates157_20 9310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan157_21

def innerTail157_19 : List Nat := 7229 :: innerTail157_20
def innerCandidates157_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates157_19_checked : candidates edge 4 941 7229 innerTail157_20 best.bound = innerCandidates157_19 := by
  unfold candidates
  have row : edge 7229 = SearchData.row7229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan157_19 : scan edge (search edge 3) 4 [941] 941 innerTail157_19 best = best := by
  rw [innerTail157_19, scan, if_neg (by decide), innerCandidates157_19_checked]
  change scan edge (search edge 3) 4 [941] 941 innerTail157_20
    (if 3 ≤ innerCandidates157_19.length then search edge 3 [941, 7229] innerCandidates157_19 8170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan157_20

def innerTail157_18 : List Nat := 6971 :: innerTail157_19
def innerCandidates157_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates157_18_checked : candidates edge 4 941 6971 innerTail157_19 best.bound = innerCandidates157_18 := by
  unfold candidates
  have row : edge 6971 = SearchData.row6971.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan157_18 : scan edge (search edge 3) 4 [941] 941 innerTail157_18 best = best := by
  rw [innerTail157_18, scan, if_neg (by decide), innerCandidates157_18_checked]
  change scan edge (search edge 3) 4 [941] 941 innerTail157_19
    (if 3 ≤ innerCandidates157_18.length then search edge 3 [941, 6971] innerCandidates157_18 7912 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan157_19

def innerTail157_17 : List Nat := 6299 :: innerTail157_18
def innerCandidates157_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates157_17_checked : candidates edge 4 941 6299 innerTail157_18 best.bound = innerCandidates157_17 := by
  unfold candidates
  have row : edge 6299 = SearchData.row6299.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan157_17 : scan edge (search edge 3) 4 [941] 941 innerTail157_17 best = best := by
  rw [innerTail157_17, scan, if_neg (by decide), innerCandidates157_17_checked]
  change scan edge (search edge 3) 4 [941] 941 innerTail157_18
    (if 3 ≤ innerCandidates157_17.length then search edge 3 [941, 6299] innerCandidates157_17 7240 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan157_18

def innerTail157_16 : List Nat := 6263 :: innerTail157_17
def innerCandidates157_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates157_16_checked : candidates edge 4 941 6263 innerTail157_17 best.bound = innerCandidates157_16 := by
  unfold candidates
  have row : edge 6263 = SearchData.row6263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan157_16 : scan edge (search edge 3) 4 [941] 941 innerTail157_16 best = best := by
  rw [innerTail157_16, scan, if_neg (by decide), innerCandidates157_16_checked]
  change scan edge (search edge 3) 4 [941] 941 innerTail157_17
    (if 3 ≤ innerCandidates157_16.length then search edge 3 [941, 6263] innerCandidates157_16 7204 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan157_17

def innerTail157_15 : List Nat := 6053 :: innerTail157_16
def innerCandidates157_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates157_15_checked : candidates edge 4 941 6053 innerTail157_16 best.bound = innerCandidates157_15 := by
  unfold candidates
  have row : edge 6053 = SearchData.row6053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan157_15 : scan edge (search edge 3) 4 [941] 941 innerTail157_15 best = best := by
  rw [innerTail157_15, scan, if_neg (by decide), innerCandidates157_15_checked]
  change scan edge (search edge 3) 4 [941] 941 innerTail157_16
    (if 3 ≤ innerCandidates157_15.length then search edge 3 [941, 6053] innerCandidates157_15 6994 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan157_16

def innerTail157_14 : List Nat := 6047 :: innerTail157_15
def innerCandidates157_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates157_14_checked : candidates edge 4 941 6047 innerTail157_15 best.bound = innerCandidates157_14 := by
  unfold candidates
  have row : edge 6047 = SearchData.row6047.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan157_14 : scan edge (search edge 3) 4 [941] 941 innerTail157_14 best = best := by
  rw [innerTail157_14, scan, if_neg (by decide), innerCandidates157_14_checked]
  change scan edge (search edge 3) 4 [941] 941 innerTail157_15
    (if 3 ≤ innerCandidates157_14.length then search edge 3 [941, 6047] innerCandidates157_14 6988 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan157_15

def innerTail157_13 : List Nat := 4817 :: innerTail157_14
def innerCandidates157_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates157_13_checked : candidates edge 4 941 4817 innerTail157_14 best.bound = innerCandidates157_13 := by
  unfold candidates
  have row : edge 4817 = SearchData.row4817.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan157_13 : scan edge (search edge 3) 4 [941] 941 innerTail157_13 best = best := by
  rw [innerTail157_13, scan, if_neg (by decide), innerCandidates157_13_checked]
  change scan edge (search edge 3) 4 [941] 941 innerTail157_14
    (if 3 ≤ innerCandidates157_13.length then search edge 3 [941, 4817] innerCandidates157_13 5758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan157_14

def innerTail157_12 : List Nat := 4649 :: innerTail157_13
def innerCandidates157_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates157_12_checked : candidates edge 4 941 4649 innerTail157_13 best.bound = innerCandidates157_12 := by
  unfold candidates
  have row : edge 4649 = SearchData.row4649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan157_12 : scan edge (search edge 3) 4 [941] 941 innerTail157_12 best = best := by
  rw [innerTail157_12, scan, if_neg (by decide), innerCandidates157_12_checked]
  change scan edge (search edge 3) 4 [941] 941 innerTail157_13
    (if 3 ≤ innerCandidates157_12.length then search edge 3 [941, 4649] innerCandidates157_12 5590 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan157_13

def innerTail157_11 : List Nat := 4409 :: innerTail157_12
def innerCandidates157_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates157_11_checked : candidates edge 4 941 4409 innerTail157_12 best.bound = innerCandidates157_11 := by
  unfold candidates
  have row : edge 4409 = SearchData.row4409.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan157_11 : scan edge (search edge 3) 4 [941] 941 innerTail157_11 best = best := by
  rw [innerTail157_11, scan, if_neg (by decide), innerCandidates157_11_checked]
  change scan edge (search edge 3) 4 [941] 941 innerTail157_12
    (if 3 ≤ innerCandidates157_11.length then search edge 3 [941, 4409] innerCandidates157_11 5350 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan157_12

def innerTail157_10 : List Nat := 3527 :: innerTail157_11
def innerCandidates157_10 : List Nat := [4649]
set_option maxRecDepth 4096 in
theorem innerCandidates157_10_checked : candidates edge 4 941 3527 innerTail157_11 best.bound = innerCandidates157_10 := by
  unfold candidates
  have row : edge 3527 = SearchData.row3527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan157_10 : scan edge (search edge 3) 4 [941] 941 innerTail157_10 best = best := by
  rw [innerTail157_10, scan, if_neg (by decide), innerCandidates157_10_checked]
  change scan edge (search edge 3) 4 [941] 941 innerTail157_11
    (if 3 ≤ innerCandidates157_10.length then search edge 3 [941, 3527] innerCandidates157_10 4468 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan157_11

def innerTail157_9 : List Nat := 3389 :: innerTail157_10
def innerCandidates157_9 : List Nat := [4409]
set_option maxRecDepth 4096 in
theorem innerCandidates157_9_checked : candidates edge 4 941 3389 innerTail157_10 best.bound = innerCandidates157_9 := by
  unfold candidates
  have row : edge 3389 = SearchData.row3389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan157_9 : scan edge (search edge 3) 4 [941] 941 innerTail157_9 best = best := by
  rw [innerTail157_9, scan, if_neg (by decide), innerCandidates157_9_checked]
  change scan edge (search edge 3) 4 [941] 941 innerTail157_10
    (if 3 ≤ innerCandidates157_9.length then search edge 3 [941, 3389] innerCandidates157_9 4330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan157_10

def innerTail157_8 : List Nat := 3221 :: innerTail157_9
def innerCandidates157_8 : List Nat := [4409, 4817]
set_option maxRecDepth 4096 in
theorem innerCandidates157_8_checked : candidates edge 4 941 3221 innerTail157_9 best.bound = innerCandidates157_8 := by
  unfold candidates
  have row : edge 3221 = SearchData.row3221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan157_8 : scan edge (search edge 3) 4 [941] 941 innerTail157_8 best = best := by
  rw [innerTail157_8, scan, if_neg (by decide), innerCandidates157_8_checked]
  change scan edge (search edge 3) 4 [941] 941 innerTail157_9
    (if 3 ≤ innerCandidates157_8.length then search edge 3 [941, 3221] innerCandidates157_8 4162 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan157_9

def innerTail157_7 : List Nat := 3137 :: innerTail157_8
def innerCandidates157_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates157_7_checked : candidates edge 4 941 3137 innerTail157_8 best.bound = innerCandidates157_7 := by
  unfold candidates
  have row : edge 3137 = SearchData.row3137.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan157_7 : scan edge (search edge 3) 4 [941] 941 innerTail157_7 best = best := by
  rw [innerTail157_7, scan, if_neg (by decide), innerCandidates157_7_checked]
  change scan edge (search edge 3) 4 [941] 941 innerTail157_8
    (if 3 ≤ innerCandidates157_7.length then search edge 3 [941, 3137] innerCandidates157_7 4078 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan157_8

def innerTail157_6 : List Nat := 2099 :: innerTail157_7
def innerCandidates157_6 : List Nat := [3527]
set_option maxRecDepth 4096 in
theorem innerCandidates157_6_checked : candidates edge 4 941 2099 innerTail157_7 best.bound = innerCandidates157_6 := by
  unfold candidates
  have row : edge 2099 = PrimePairCache.smallRow2099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan157_6 : scan edge (search edge 3) 4 [941] 941 innerTail157_6 best = best := by
  rw [innerTail157_6, scan, if_neg (by decide), innerCandidates157_6_checked]
  change scan edge (search edge 3) 4 [941] 941 innerTail157_7
    (if 3 ≤ innerCandidates157_6.length then search edge 3 [941, 2099] innerCandidates157_6 3040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan157_7

def innerTail157_5 : List Nat := 1709 :: innerTail157_6
def innerCandidates157_5 : List Nat := [3137, 3389]
set_option maxRecDepth 4096 in
theorem innerCandidates157_5_checked : candidates edge 4 941 1709 innerTail157_6 best.bound = innerCandidates157_5 := by
  unfold candidates
  have row : edge 1709 = PrimePairCache.smallRow1709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan157_5 : scan edge (search edge 3) 4 [941] 941 innerTail157_5 best = best := by
  rw [innerTail157_5, scan, if_neg (by decide), innerCandidates157_5_checked]
  change scan edge (search edge 3) 4 [941] 941 innerTail157_6
    (if 3 ≤ innerCandidates157_5.length then search edge 3 [941, 1709] innerCandidates157_5 2650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan157_6

def innerTail157_4 : List Nat := 1607 :: innerTail157_5
def innerCandidates157_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates157_4_checked : candidates edge 4 941 1607 innerTail157_5 best.bound = innerCandidates157_4 := by
  unfold candidates
  have row : edge 1607 = PrimePairCache.smallRow1607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan157_4 : scan edge (search edge 3) 4 [941] 941 innerTail157_4 best = best := by
  rw [innerTail157_4, scan, if_neg (by decide), innerCandidates157_4_checked]
  change scan edge (search edge 3) 4 [941] 941 innerTail157_5
    (if 3 ≤ innerCandidates157_4.length then search edge 3 [941, 1607] innerCandidates157_4 2548 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan157_5

def innerTail157_3 : List Nat := 1511 :: innerTail157_4
def innerCandidates157_3 : List Nat := [3389]
set_option maxRecDepth 4096 in
theorem innerCandidates157_3_checked : candidates edge 4 941 1511 innerTail157_4 best.bound = innerCandidates157_3 := by
  unfold candidates
  have row : edge 1511 = PrimePairCache.smallRow1511.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan157_3 : scan edge (search edge 3) 4 [941] 941 innerTail157_3 best = best := by
  rw [innerTail157_3, scan, if_neg (by decide), innerCandidates157_3_checked]
  change scan edge (search edge 3) 4 [941] 941 innerTail157_4
    (if 3 ≤ innerCandidates157_3.length then search edge 3 [941, 1511] innerCandidates157_3 2452 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan157_4

def innerTail157_2 : List Nat := 1229 :: innerTail157_3
def innerCandidates157_2 : List Nat := [1607, 3389]
set_option maxRecDepth 4096 in
theorem innerCandidates157_2_checked : candidates edge 4 941 1229 innerTail157_3 best.bound = innerCandidates157_2 := by
  unfold candidates
  have row : edge 1229 = PrimePairCache.smallRow1229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan157_2 : scan edge (search edge 3) 4 [941] 941 innerTail157_2 best = best := by
  rw [innerTail157_2, scan, if_neg (by decide), innerCandidates157_2_checked]
  change scan edge (search edge 3) 4 [941] 941 innerTail157_3
    (if 3 ≤ innerCandidates157_2.length then search edge 3 [941, 1229] innerCandidates157_2 2170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan157_3

def innerTail157_1 : List Nat := 1187 :: innerTail157_2
def innerCandidates157_1 : List Nat := [4649]
set_option maxRecDepth 4096 in
theorem innerCandidates157_1_checked : candidates edge 4 941 1187 innerTail157_2 best.bound = innerCandidates157_1 := by
  unfold candidates
  have row : edge 1187 = PrimePairCache.smallRow1187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan157_1 : scan edge (search edge 3) 4 [941] 941 innerTail157_1 best = best := by
  rw [innerTail157_1, scan, if_neg (by decide), innerCandidates157_1_checked]
  change scan edge (search edge 3) 4 [941] 941 innerTail157_2
    (if 3 ≤ innerCandidates157_1.length then search edge 3 [941, 1187] innerCandidates157_1 2128 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan157_2

def innerTail157_0 : List Nat := 1049 :: innerTail157_1
def innerCandidates157_0 : List Nat := [6971, 7229]
set_option maxRecDepth 4096 in
theorem innerCandidates157_0_checked : candidates edge 4 941 1049 innerTail157_1 best.bound = innerCandidates157_0 := by
  unfold candidates
  have row : edge 1049 = PrimePairCache.smallRow1049.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan157_0 : scan edge (search edge 3) 4 [941] 941 innerTail157_0 best = best := by
  rw [innerTail157_0, scan, if_neg (by decide), innerCandidates157_0_checked]
  change scan edge (search edge 3) 4 [941] 941 innerTail157_1
    (if 3 ≤ innerCandidates157_0.length then search edge 3 [941, 1049] innerCandidates157_0 1990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan157_1

theorem rootBranch157 : search edge 4 [941] rootCandidates157 941 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [941] 941 innerTail157_0 best = best
  exact innerScan157_0

end ProjectEulerSolutions.PrimePairUpper.Search
