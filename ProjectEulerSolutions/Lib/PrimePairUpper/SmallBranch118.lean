import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates328 : List Nat := [2389, 3049, 3847, 4111, 4201, 4513, 4519, 4663, 4909, 5527, 5581, 5737, 5839, 6043, 6451, 6673, 7639, 7681, 7873, 8059]
set_option maxRecDepth 4096 in
theorem rootCandidates328_checked : candidates edge 5 0 2221 tail329 best.bound = rootCandidates328 := by
  unfold candidates
  have row : edge 2221 = PrimePairCache.smallRow2221.contains := by funext q; rfl
  rw [row]
  decide

def innerTail328_20 : List Nat := []
theorem innerScan328_20 : scan edge (search edge 3) 4 [2221] 2221 innerTail328_20 best = best := rfl

def innerTail328_19 : List Nat := 8059 :: innerTail328_20
def innerCandidates328_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates328_19_checked : candidates edge 4 2221 8059 innerTail328_20 best.bound = innerCandidates328_19 := by
  unfold candidates
  have row : edge 8059 = SearchData.row8059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan328_19 : scan edge (search edge 3) 4 [2221] 2221 innerTail328_19 best = best := by
  rw [innerTail328_19, scan, if_neg (by decide), innerCandidates328_19_checked]
  change scan edge (search edge 3) 4 [2221] 2221 innerTail328_20
    (if 3 ≤ innerCandidates328_19.length then search edge 3 [2221, 8059] innerCandidates328_19 10280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan328_20

def innerTail328_18 : List Nat := 7873 :: innerTail328_19
def innerCandidates328_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates328_18_checked : candidates edge 4 2221 7873 innerTail328_19 best.bound = innerCandidates328_18 := by
  unfold candidates
  have row : edge 7873 = SearchData.row7873.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan328_18 : scan edge (search edge 3) 4 [2221] 2221 innerTail328_18 best = best := by
  rw [innerTail328_18, scan, if_neg (by decide), innerCandidates328_18_checked]
  change scan edge (search edge 3) 4 [2221] 2221 innerTail328_19
    (if 3 ≤ innerCandidates328_18.length then search edge 3 [2221, 7873] innerCandidates328_18 10094 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan328_19

def innerTail328_17 : List Nat := 7681 :: innerTail328_18
def innerCandidates328_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates328_17_checked : candidates edge 4 2221 7681 innerTail328_18 best.bound = innerCandidates328_17 := by
  unfold candidates
  have row : edge 7681 = SearchData.row7681.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan328_17 : scan edge (search edge 3) 4 [2221] 2221 innerTail328_17 best = best := by
  rw [innerTail328_17, scan, if_neg (by decide), innerCandidates328_17_checked]
  change scan edge (search edge 3) 4 [2221] 2221 innerTail328_18
    (if 3 ≤ innerCandidates328_17.length then search edge 3 [2221, 7681] innerCandidates328_17 9902 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan328_18

def innerTail328_16 : List Nat := 7639 :: innerTail328_17
def innerCandidates328_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates328_16_checked : candidates edge 4 2221 7639 innerTail328_17 best.bound = innerCandidates328_16 := by
  unfold candidates
  have row : edge 7639 = SearchData.row7639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan328_16 : scan edge (search edge 3) 4 [2221] 2221 innerTail328_16 best = best := by
  rw [innerTail328_16, scan, if_neg (by decide), innerCandidates328_16_checked]
  change scan edge (search edge 3) 4 [2221] 2221 innerTail328_17
    (if 3 ≤ innerCandidates328_16.length then search edge 3 [2221, 7639] innerCandidates328_16 9860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan328_17

def innerTail328_15 : List Nat := 6673 :: innerTail328_16
def innerCandidates328_15 : List Nat := [7639]
set_option maxRecDepth 4096 in
theorem innerCandidates328_15_checked : candidates edge 4 2221 6673 innerTail328_16 best.bound = innerCandidates328_15 := by
  unfold candidates
  have row : edge 6673 = SearchData.row6673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan328_15 : scan edge (search edge 3) 4 [2221] 2221 innerTail328_15 best = best := by
  rw [innerTail328_15, scan, if_neg (by decide), innerCandidates328_15_checked]
  change scan edge (search edge 3) 4 [2221] 2221 innerTail328_16
    (if 3 ≤ innerCandidates328_15.length then search edge 3 [2221, 6673] innerCandidates328_15 8894 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan328_16

def innerTail328_14 : List Nat := 6451 :: innerTail328_15
def innerCandidates328_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates328_14_checked : candidates edge 4 2221 6451 innerTail328_15 best.bound = innerCandidates328_14 := by
  unfold candidates
  have row : edge 6451 = SearchData.row6451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan328_14 : scan edge (search edge 3) 4 [2221] 2221 innerTail328_14 best = best := by
  rw [innerTail328_14, scan, if_neg (by decide), innerCandidates328_14_checked]
  change scan edge (search edge 3) 4 [2221] 2221 innerTail328_15
    (if 3 ≤ innerCandidates328_14.length then search edge 3 [2221, 6451] innerCandidates328_14 8672 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan328_15

def innerTail328_13 : List Nat := 6043 :: innerTail328_14
def innerCandidates328_13 : List Nat := [6451]
set_option maxRecDepth 4096 in
theorem innerCandidates328_13_checked : candidates edge 4 2221 6043 innerTail328_14 best.bound = innerCandidates328_13 := by
  unfold candidates
  have row : edge 6043 = SearchData.row6043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan328_13 : scan edge (search edge 3) 4 [2221] 2221 innerTail328_13 best = best := by
  rw [innerTail328_13, scan, if_neg (by decide), innerCandidates328_13_checked]
  change scan edge (search edge 3) 4 [2221] 2221 innerTail328_14
    (if 3 ≤ innerCandidates328_13.length then search edge 3 [2221, 6043] innerCandidates328_13 8264 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan328_14

def innerTail328_12 : List Nat := 5839 :: innerTail328_13
def innerCandidates328_12 : List Nat := [7681]
set_option maxRecDepth 4096 in
theorem innerCandidates328_12_checked : candidates edge 4 2221 5839 innerTail328_13 best.bound = innerCandidates328_12 := by
  unfold candidates
  have row : edge 5839 = SearchData.row5839.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan328_12 : scan edge (search edge 3) 4 [2221] 2221 innerTail328_12 best = best := by
  rw [innerTail328_12, scan, if_neg (by decide), innerCandidates328_12_checked]
  change scan edge (search edge 3) 4 [2221] 2221 innerTail328_13
    (if 3 ≤ innerCandidates328_12.length then search edge 3 [2221, 5839] innerCandidates328_12 8060 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan328_13

def innerTail328_11 : List Nat := 5737 :: innerTail328_12
def innerCandidates328_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates328_11_checked : candidates edge 4 2221 5737 innerTail328_12 best.bound = innerCandidates328_11 := by
  unfold candidates
  have row : edge 5737 = SearchData.row5737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan328_11 : scan edge (search edge 3) 4 [2221] 2221 innerTail328_11 best = best := by
  rw [innerTail328_11, scan, if_neg (by decide), innerCandidates328_11_checked]
  change scan edge (search edge 3) 4 [2221] 2221 innerTail328_12
    (if 3 ≤ innerCandidates328_11.length then search edge 3 [2221, 5737] innerCandidates328_11 7958 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan328_12

def innerTail328_10 : List Nat := 5581 :: innerTail328_11
def innerCandidates328_10 : List Nat := [5737, 5839]
set_option maxRecDepth 4096 in
theorem innerCandidates328_10_checked : candidates edge 4 2221 5581 innerTail328_11 best.bound = innerCandidates328_10 := by
  unfold candidates
  have row : edge 5581 = SearchData.row5581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan328_10 : scan edge (search edge 3) 4 [2221] 2221 innerTail328_10 best = best := by
  rw [innerTail328_10, scan, if_neg (by decide), innerCandidates328_10_checked]
  change scan edge (search edge 3) 4 [2221] 2221 innerTail328_11
    (if 3 ≤ innerCandidates328_10.length then search edge 3 [2221, 5581] innerCandidates328_10 7802 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan328_11

def innerTail328_9 : List Nat := 5527 :: innerTail328_10
def innerCandidates328_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates328_9_checked : candidates edge 4 2221 5527 innerTail328_10 best.bound = innerCandidates328_9 := by
  unfold candidates
  have row : edge 5527 = SearchData.row5527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan328_9 : scan edge (search edge 3) 4 [2221] 2221 innerTail328_9 best = best := by
  rw [innerTail328_9, scan, if_neg (by decide), innerCandidates328_9_checked]
  change scan edge (search edge 3) 4 [2221] 2221 innerTail328_10
    (if 3 ≤ innerCandidates328_9.length then search edge 3 [2221, 5527] innerCandidates328_9 7748 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan328_10

def innerTail328_8 : List Nat := 4909 :: innerTail328_9
def innerCandidates328_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates328_8_checked : candidates edge 4 2221 4909 innerTail328_9 best.bound = innerCandidates328_8 := by
  unfold candidates
  have row : edge 4909 = SearchData.row4909.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan328_8 : scan edge (search edge 3) 4 [2221] 2221 innerTail328_8 best = best := by
  rw [innerTail328_8, scan, if_neg (by decide), innerCandidates328_8_checked]
  change scan edge (search edge 3) 4 [2221] 2221 innerTail328_9
    (if 3 ≤ innerCandidates328_8.length then search edge 3 [2221, 4909] innerCandidates328_8 7130 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan328_9

def innerTail328_7 : List Nat := 4663 :: innerTail328_8
def innerCandidates328_7 : List Nat := [5839]
set_option maxRecDepth 4096 in
theorem innerCandidates328_7_checked : candidates edge 4 2221 4663 innerTail328_8 best.bound = innerCandidates328_7 := by
  unfold candidates
  have row : edge 4663 = SearchData.row4663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan328_7 : scan edge (search edge 3) 4 [2221] 2221 innerTail328_7 best = best := by
  rw [innerTail328_7, scan, if_neg (by decide), innerCandidates328_7_checked]
  change scan edge (search edge 3) 4 [2221] 2221 innerTail328_8
    (if 3 ≤ innerCandidates328_7.length then search edge 3 [2221, 4663] innerCandidates328_7 6884 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan328_8

def innerTail328_6 : List Nat := 4519 :: innerTail328_7
def innerCandidates328_6 : List Nat := [6043]
set_option maxRecDepth 4096 in
theorem innerCandidates328_6_checked : candidates edge 4 2221 4519 innerTail328_7 best.bound = innerCandidates328_6 := by
  unfold candidates
  have row : edge 4519 = SearchData.row4519.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan328_6 : scan edge (search edge 3) 4 [2221] 2221 innerTail328_6 best = best := by
  rw [innerTail328_6, scan, if_neg (by decide), innerCandidates328_6_checked]
  change scan edge (search edge 3) 4 [2221] 2221 innerTail328_7
    (if 3 ≤ innerCandidates328_6.length then search edge 3 [2221, 4519] innerCandidates328_6 6740 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan328_7

def innerTail328_5 : List Nat := 4513 :: innerTail328_6
def innerCandidates328_5 : List Nat := [5581]
set_option maxRecDepth 4096 in
theorem innerCandidates328_5_checked : candidates edge 4 2221 4513 innerTail328_6 best.bound = innerCandidates328_5 := by
  unfold candidates
  have row : edge 4513 = SearchData.row4513.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan328_5 : scan edge (search edge 3) 4 [2221] 2221 innerTail328_5 best = best := by
  rw [innerTail328_5, scan, if_neg (by decide), innerCandidates328_5_checked]
  change scan edge (search edge 3) 4 [2221] 2221 innerTail328_6
    (if 3 ≤ innerCandidates328_5.length then search edge 3 [2221, 4513] innerCandidates328_5 6734 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan328_6

def innerTail328_4 : List Nat := 4201 :: innerTail328_5
def innerCandidates328_4 : List Nat := [5527]
set_option maxRecDepth 4096 in
theorem innerCandidates328_4_checked : candidates edge 4 2221 4201 innerTail328_5 best.bound = innerCandidates328_4 := by
  unfold candidates
  have row : edge 4201 = SearchData.row4201.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan328_4 : scan edge (search edge 3) 4 [2221] 2221 innerTail328_4 best = best := by
  rw [innerTail328_4, scan, if_neg (by decide), innerCandidates328_4_checked]
  change scan edge (search edge 3) 4 [2221] 2221 innerTail328_5
    (if 3 ≤ innerCandidates328_4.length then search edge 3 [2221, 4201] innerCandidates328_4 6422 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan328_5

def innerTail328_3 : List Nat := 4111 :: innerTail328_4
def innerCandidates328_3 : List Nat := [5527]
set_option maxRecDepth 4096 in
theorem innerCandidates328_3_checked : candidates edge 4 2221 4111 innerTail328_4 best.bound = innerCandidates328_3 := by
  unfold candidates
  have row : edge 4111 = SearchData.row4111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan328_3 : scan edge (search edge 3) 4 [2221] 2221 innerTail328_3 best = best := by
  rw [innerTail328_3, scan, if_neg (by decide), innerCandidates328_3_checked]
  change scan edge (search edge 3) 4 [2221] 2221 innerTail328_4
    (if 3 ≤ innerCandidates328_3.length then search edge 3 [2221, 4111] innerCandidates328_3 6332 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan328_4

def innerTail328_2 : List Nat := 3847 :: innerTail328_3
def innerCandidates328_2 : List Nat := [5737]
set_option maxRecDepth 4096 in
theorem innerCandidates328_2_checked : candidates edge 4 2221 3847 innerTail328_3 best.bound = innerCandidates328_2 := by
  unfold candidates
  have row : edge 3847 = SearchData.row3847.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan328_2 : scan edge (search edge 3) 4 [2221] 2221 innerTail328_2 best = best := by
  rw [innerTail328_2, scan, if_neg (by decide), innerCandidates328_2_checked]
  change scan edge (search edge 3) 4 [2221] 2221 innerTail328_3
    (if 3 ≤ innerCandidates328_2.length then search edge 3 [2221, 3847] innerCandidates328_2 6068 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan328_3

def innerTail328_1 : List Nat := 3049 :: innerTail328_2
def innerCandidates328_1 : List Nat := [5581]
set_option maxRecDepth 4096 in
theorem innerCandidates328_1_checked : candidates edge 4 2221 3049 innerTail328_2 best.bound = innerCandidates328_1 := by
  unfold candidates
  have row : edge 3049 = SearchData.row3049.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan328_1 : scan edge (search edge 3) 4 [2221] 2221 innerTail328_1 best = best := by
  rw [innerTail328_1, scan, if_neg (by decide), innerCandidates328_1_checked]
  change scan edge (search edge 3) 4 [2221] 2221 innerTail328_2
    (if 3 ≤ innerCandidates328_1.length then search edge 3 [2221, 3049] innerCandidates328_1 5270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan328_2

def innerTail328_0 : List Nat := 2389 :: innerTail328_1
def innerCandidates328_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates328_0_checked : candidates edge 4 2221 2389 innerTail328_1 best.bound = innerCandidates328_0 := by
  unfold candidates
  have row : edge 2389 = PrimePairCache.smallRow2389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan328_0 : scan edge (search edge 3) 4 [2221] 2221 innerTail328_0 best = best := by
  rw [innerTail328_0, scan, if_neg (by decide), innerCandidates328_0_checked]
  change scan edge (search edge 3) 4 [2221] 2221 innerTail328_1
    (if 3 ≤ innerCandidates328_0.length then search edge 3 [2221, 2389] innerCandidates328_0 4610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan328_1

theorem rootBranch328 : search edge 4 [2221] rootCandidates328 2221 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2221] 2221 innerTail328_0 best = best
  exact innerScan328_0

def rootCandidates327 : List Nat := [2711, 3581, 4253, 4451, 4691, 5417, 6029, 7457, 7547, 8093, 8147, 8243]
set_option maxRecDepth 4096 in
theorem rootCandidates327_checked : candidates edge 5 0 2213 tail328 best.bound = rootCandidates327 := by
  unfold candidates
  have row : edge 2213 = PrimePairCache.smallRow2213.contains := by funext q; rfl
  rw [row]
  decide

def innerTail327_12 : List Nat := []
theorem innerScan327_12 : scan edge (search edge 3) 4 [2213] 2213 innerTail327_12 best = best := rfl

def innerTail327_11 : List Nat := 8243 :: innerTail327_12
def innerCandidates327_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates327_11_checked : candidates edge 4 2213 8243 innerTail327_12 best.bound = innerCandidates327_11 := by
  unfold candidates
  have row : edge 8243 = SearchData.row8243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan327_11 : scan edge (search edge 3) 4 [2213] 2213 innerTail327_11 best = best := by
  rw [innerTail327_11, scan, if_neg (by decide), innerCandidates327_11_checked]
  change scan edge (search edge 3) 4 [2213] 2213 innerTail327_12
    (if 3 ≤ innerCandidates327_11.length then search edge 3 [2213, 8243] innerCandidates327_11 10456 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan327_12

def innerTail327_10 : List Nat := 8147 :: innerTail327_11
def innerCandidates327_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates327_10_checked : candidates edge 4 2213 8147 innerTail327_11 best.bound = innerCandidates327_10 := by
  unfold candidates
  have row : edge 8147 = SearchData.row8147.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan327_10 : scan edge (search edge 3) 4 [2213] 2213 innerTail327_10 best = best := by
  rw [innerTail327_10, scan, if_neg (by decide), innerCandidates327_10_checked]
  change scan edge (search edge 3) 4 [2213] 2213 innerTail327_11
    (if 3 ≤ innerCandidates327_10.length then search edge 3 [2213, 8147] innerCandidates327_10 10360 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan327_11

def innerTail327_9 : List Nat := 8093 :: innerTail327_10
def innerCandidates327_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates327_9_checked : candidates edge 4 2213 8093 innerTail327_10 best.bound = innerCandidates327_9 := by
  unfold candidates
  have row : edge 8093 = SearchData.row8093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan327_9 : scan edge (search edge 3) 4 [2213] 2213 innerTail327_9 best = best := by
  rw [innerTail327_9, scan, if_neg (by decide), innerCandidates327_9_checked]
  change scan edge (search edge 3) 4 [2213] 2213 innerTail327_10
    (if 3 ≤ innerCandidates327_9.length then search edge 3 [2213, 8093] innerCandidates327_9 10306 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan327_10

def innerTail327_8 : List Nat := 7547 :: innerTail327_9
def innerCandidates327_8 : List Nat := [8093]
set_option maxRecDepth 4096 in
theorem innerCandidates327_8_checked : candidates edge 4 2213 7547 innerTail327_9 best.bound = innerCandidates327_8 := by
  unfold candidates
  have row : edge 7547 = SearchData.row7547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan327_8 : scan edge (search edge 3) 4 [2213] 2213 innerTail327_8 best = best := by
  rw [innerTail327_8, scan, if_neg (by decide), innerCandidates327_8_checked]
  change scan edge (search edge 3) 4 [2213] 2213 innerTail327_9
    (if 3 ≤ innerCandidates327_8.length then search edge 3 [2213, 7547] innerCandidates327_8 9760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan327_9

def innerTail327_7 : List Nat := 7457 :: innerTail327_8
def innerCandidates327_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates327_7_checked : candidates edge 4 2213 7457 innerTail327_8 best.bound = innerCandidates327_7 := by
  unfold candidates
  have row : edge 7457 = SearchData.row7457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan327_7 : scan edge (search edge 3) 4 [2213] 2213 innerTail327_7 best = best := by
  rw [innerTail327_7, scan, if_neg (by decide), innerCandidates327_7_checked]
  change scan edge (search edge 3) 4 [2213] 2213 innerTail327_8
    (if 3 ≤ innerCandidates327_7.length then search edge 3 [2213, 7457] innerCandidates327_7 9670 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan327_8

def innerTail327_6 : List Nat := 6029 :: innerTail327_7
def innerCandidates327_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates327_6_checked : candidates edge 4 2213 6029 innerTail327_7 best.bound = innerCandidates327_6 := by
  unfold candidates
  have row : edge 6029 = SearchData.row6029.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan327_6 : scan edge (search edge 3) 4 [2213] 2213 innerTail327_6 best = best := by
  rw [innerTail327_6, scan, if_neg (by decide), innerCandidates327_6_checked]
  change scan edge (search edge 3) 4 [2213] 2213 innerTail327_7
    (if 3 ≤ innerCandidates327_6.length then search edge 3 [2213, 6029] innerCandidates327_6 8242 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan327_7

def innerTail327_5 : List Nat := 5417 :: innerTail327_6
def innerCandidates327_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates327_5_checked : candidates edge 4 2213 5417 innerTail327_6 best.bound = innerCandidates327_5 := by
  unfold candidates
  have row : edge 5417 = SearchData.row5417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan327_5 : scan edge (search edge 3) 4 [2213] 2213 innerTail327_5 best = best := by
  rw [innerTail327_5, scan, if_neg (by decide), innerCandidates327_5_checked]
  change scan edge (search edge 3) 4 [2213] 2213 innerTail327_6
    (if 3 ≤ innerCandidates327_5.length then search edge 3 [2213, 5417] innerCandidates327_5 7630 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan327_6

def innerTail327_4 : List Nat := 4691 :: innerTail327_5
def innerCandidates327_4 : List Nat := [7457]
set_option maxRecDepth 4096 in
theorem innerCandidates327_4_checked : candidates edge 4 2213 4691 innerTail327_5 best.bound = innerCandidates327_4 := by
  unfold candidates
  have row : edge 4691 = SearchData.row4691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan327_4 : scan edge (search edge 3) 4 [2213] 2213 innerTail327_4 best = best := by
  rw [innerTail327_4, scan, if_neg (by decide), innerCandidates327_4_checked]
  change scan edge (search edge 3) 4 [2213] 2213 innerTail327_5
    (if 3 ≤ innerCandidates327_4.length then search edge 3 [2213, 4691] innerCandidates327_4 6904 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan327_5

def innerTail327_3 : List Nat := 4451 :: innerTail327_4
def innerCandidates327_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates327_3_checked : candidates edge 4 2213 4451 innerTail327_4 best.bound = innerCandidates327_3 := by
  unfold candidates
  have row : edge 4451 = SearchData.row4451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan327_3 : scan edge (search edge 3) 4 [2213] 2213 innerTail327_3 best = best := by
  rw [innerTail327_3, scan, if_neg (by decide), innerCandidates327_3_checked]
  change scan edge (search edge 3) 4 [2213] 2213 innerTail327_4
    (if 3 ≤ innerCandidates327_3.length then search edge 3 [2213, 4451] innerCandidates327_3 6664 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan327_4

def innerTail327_2 : List Nat := 4253 :: innerTail327_3
def innerCandidates327_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates327_2_checked : candidates edge 4 2213 4253 innerTail327_3 best.bound = innerCandidates327_2 := by
  unfold candidates
  have row : edge 4253 = SearchData.row4253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan327_2 : scan edge (search edge 3) 4 [2213] 2213 innerTail327_2 best = best := by
  rw [innerTail327_2, scan, if_neg (by decide), innerCandidates327_2_checked]
  change scan edge (search edge 3) 4 [2213] 2213 innerTail327_3
    (if 3 ≤ innerCandidates327_2.length then search edge 3 [2213, 4253] innerCandidates327_2 6466 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan327_3

def innerTail327_1 : List Nat := 3581 :: innerTail327_2
def innerCandidates327_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates327_1_checked : candidates edge 4 2213 3581 innerTail327_2 best.bound = innerCandidates327_1 := by
  unfold candidates
  have row : edge 3581 = SearchData.row3581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan327_1 : scan edge (search edge 3) 4 [2213] 2213 innerTail327_1 best = best := by
  rw [innerTail327_1, scan, if_neg (by decide), innerCandidates327_1_checked]
  change scan edge (search edge 3) 4 [2213] 2213 innerTail327_2
    (if 3 ≤ innerCandidates327_1.length then search edge 3 [2213, 3581] innerCandidates327_1 5794 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan327_2

def innerTail327_0 : List Nat := 2711 :: innerTail327_1
def innerCandidates327_0 : List Nat := [7457]
set_option maxRecDepth 4096 in
theorem innerCandidates327_0_checked : candidates edge 4 2213 2711 innerTail327_1 best.bound = innerCandidates327_0 := by
  unfold candidates
  have row : edge 2711 = SearchData.row2711.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan327_0 : scan edge (search edge 3) 4 [2213] 2213 innerTail327_0 best = best := by
  rw [innerTail327_0, scan, if_neg (by decide), innerCandidates327_0_checked]
  change scan edge (search edge 3) 4 [2213] 2213 innerTail327_1
    (if 3 ≤ innerCandidates327_0.length then search edge 3 [2213, 2711] innerCandidates327_0 4924 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan327_1

theorem rootBranch327 : search edge 4 [2213] rootCandidates327 2213 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2213] 2213 innerTail327_0 best = best
  exact innerScan327_0

def rootCandidates326 : List Nat := [2339, 2753, 2777, 2819, 3347, 3929, 4271, 4337, 5171, 6053, 6563, 7193, 8363]
set_option maxRecDepth 4096 in
theorem rootCandidates326_checked : candidates edge 5 0 2207 tail327 best.bound = rootCandidates326 := by
  unfold candidates
  have row : edge 2207 = PrimePairCache.smallRow2207.contains := by funext q; rfl
  rw [row]
  decide

def innerTail326_13 : List Nat := []
theorem innerScan326_13 : scan edge (search edge 3) 4 [2207] 2207 innerTail326_13 best = best := rfl

def innerTail326_12 : List Nat := 8363 :: innerTail326_13
def innerCandidates326_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates326_12_checked : candidates edge 4 2207 8363 innerTail326_13 best.bound = innerCandidates326_12 := by
  unfold candidates
  have row : edge 8363 = SearchData.row8363.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan326_12 : scan edge (search edge 3) 4 [2207] 2207 innerTail326_12 best = best := by
  rw [innerTail326_12, scan, if_neg (by decide), innerCandidates326_12_checked]
  change scan edge (search edge 3) 4 [2207] 2207 innerTail326_13
    (if 3 ≤ innerCandidates326_12.length then search edge 3 [2207, 8363] innerCandidates326_12 10570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan326_13

def innerTail326_11 : List Nat := 7193 :: innerTail326_12
def innerCandidates326_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates326_11_checked : candidates edge 4 2207 7193 innerTail326_12 best.bound = innerCandidates326_11 := by
  unfold candidates
  have row : edge 7193 = SearchData.row7193.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan326_11 : scan edge (search edge 3) 4 [2207] 2207 innerTail326_11 best = best := by
  rw [innerTail326_11, scan, if_neg (by decide), innerCandidates326_11_checked]
  change scan edge (search edge 3) 4 [2207] 2207 innerTail326_12
    (if 3 ≤ innerCandidates326_11.length then search edge 3 [2207, 7193] innerCandidates326_11 9400 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan326_12

def innerTail326_10 : List Nat := 6563 :: innerTail326_11
def innerCandidates326_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates326_10_checked : candidates edge 4 2207 6563 innerTail326_11 best.bound = innerCandidates326_10 := by
  unfold candidates
  have row : edge 6563 = SearchData.row6563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan326_10 : scan edge (search edge 3) 4 [2207] 2207 innerTail326_10 best = best := by
  rw [innerTail326_10, scan, if_neg (by decide), innerCandidates326_10_checked]
  change scan edge (search edge 3) 4 [2207] 2207 innerTail326_11
    (if 3 ≤ innerCandidates326_10.length then search edge 3 [2207, 6563] innerCandidates326_10 8770 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan326_11

def innerTail326_9 : List Nat := 6053 :: innerTail326_10
def innerCandidates326_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates326_9_checked : candidates edge 4 2207 6053 innerTail326_10 best.bound = innerCandidates326_9 := by
  unfold candidates
  have row : edge 6053 = SearchData.row6053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan326_9 : scan edge (search edge 3) 4 [2207] 2207 innerTail326_9 best = best := by
  rw [innerTail326_9, scan, if_neg (by decide), innerCandidates326_9_checked]
  change scan edge (search edge 3) 4 [2207] 2207 innerTail326_10
    (if 3 ≤ innerCandidates326_9.length then search edge 3 [2207, 6053] innerCandidates326_9 8260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan326_10

def innerTail326_8 : List Nat := 5171 :: innerTail326_9
def innerCandidates326_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates326_8_checked : candidates edge 4 2207 5171 innerTail326_9 best.bound = innerCandidates326_8 := by
  unfold candidates
  have row : edge 5171 = SearchData.row5171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan326_8 : scan edge (search edge 3) 4 [2207] 2207 innerTail326_8 best = best := by
  rw [innerTail326_8, scan, if_neg (by decide), innerCandidates326_8_checked]
  change scan edge (search edge 3) 4 [2207] 2207 innerTail326_9
    (if 3 ≤ innerCandidates326_8.length then search edge 3 [2207, 5171] innerCandidates326_8 7378 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan326_9

def innerTail326_7 : List Nat := 4337 :: innerTail326_8
def innerCandidates326_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates326_7_checked : candidates edge 4 2207 4337 innerTail326_8 best.bound = innerCandidates326_7 := by
  unfold candidates
  have row : edge 4337 = SearchData.row4337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan326_7 : scan edge (search edge 3) 4 [2207] 2207 innerTail326_7 best = best := by
  rw [innerTail326_7, scan, if_neg (by decide), innerCandidates326_7_checked]
  change scan edge (search edge 3) 4 [2207] 2207 innerTail326_8
    (if 3 ≤ innerCandidates326_7.length then search edge 3 [2207, 4337] innerCandidates326_7 6544 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan326_8

def innerTail326_6 : List Nat := 4271 :: innerTail326_7
def innerCandidates326_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates326_6_checked : candidates edge 4 2207 4271 innerTail326_7 best.bound = innerCandidates326_6 := by
  unfold candidates
  have row : edge 4271 = SearchData.row4271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan326_6 : scan edge (search edge 3) 4 [2207] 2207 innerTail326_6 best = best := by
  rw [innerTail326_6, scan, if_neg (by decide), innerCandidates326_6_checked]
  change scan edge (search edge 3) 4 [2207] 2207 innerTail326_7
    (if 3 ≤ innerCandidates326_6.length then search edge 3 [2207, 4271] innerCandidates326_6 6478 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan326_7

def innerTail326_5 : List Nat := 3929 :: innerTail326_6
def innerCandidates326_5 : List Nat := [6053]
set_option maxRecDepth 4096 in
theorem innerCandidates326_5_checked : candidates edge 4 2207 3929 innerTail326_6 best.bound = innerCandidates326_5 := by
  unfold candidates
  have row : edge 3929 = SearchData.row3929.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan326_5 : scan edge (search edge 3) 4 [2207] 2207 innerTail326_5 best = best := by
  rw [innerTail326_5, scan, if_neg (by decide), innerCandidates326_5_checked]
  change scan edge (search edge 3) 4 [2207] 2207 innerTail326_6
    (if 3 ≤ innerCandidates326_5.length then search edge 3 [2207, 3929] innerCandidates326_5 6136 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan326_6

def innerTail326_4 : List Nat := 3347 :: innerTail326_5
def innerCandidates326_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates326_4_checked : candidates edge 4 2207 3347 innerTail326_5 best.bound = innerCandidates326_4 := by
  unfold candidates
  have row : edge 3347 = SearchData.row3347.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan326_4 : scan edge (search edge 3) 4 [2207] 2207 innerTail326_4 best = best := by
  rw [innerTail326_4, scan, if_neg (by decide), innerCandidates326_4_checked]
  change scan edge (search edge 3) 4 [2207] 2207 innerTail326_5
    (if 3 ≤ innerCandidates326_4.length then search edge 3 [2207, 3347] innerCandidates326_4 5554 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan326_5

def innerTail326_3 : List Nat := 2819 :: innerTail326_4
def innerCandidates326_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates326_3_checked : candidates edge 4 2207 2819 innerTail326_4 best.bound = innerCandidates326_3 := by
  unfold candidates
  have row : edge 2819 = SearchData.row2819.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan326_3 : scan edge (search edge 3) 4 [2207] 2207 innerTail326_3 best = best := by
  rw [innerTail326_3, scan, if_neg (by decide), innerCandidates326_3_checked]
  change scan edge (search edge 3) 4 [2207] 2207 innerTail326_4
    (if 3 ≤ innerCandidates326_3.length then search edge 3 [2207, 2819] innerCandidates326_3 5026 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan326_4

def innerTail326_2 : List Nat := 2777 :: innerTail326_3
def innerCandidates326_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates326_2_checked : candidates edge 4 2207 2777 innerTail326_3 best.bound = innerCandidates326_2 := by
  unfold candidates
  have row : edge 2777 = SearchData.row2777.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan326_2 : scan edge (search edge 3) 4 [2207] 2207 innerTail326_2 best = best := by
  rw [innerTail326_2, scan, if_neg (by decide), innerCandidates326_2_checked]
  change scan edge (search edge 3) 4 [2207] 2207 innerTail326_3
    (if 3 ≤ innerCandidates326_2.length then search edge 3 [2207, 2777] innerCandidates326_2 4984 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan326_3

def innerTail326_1 : List Nat := 2753 :: innerTail326_2
def innerCandidates326_1 : List Nat := [5171]
set_option maxRecDepth 4096 in
theorem innerCandidates326_1_checked : candidates edge 4 2207 2753 innerTail326_2 best.bound = innerCandidates326_1 := by
  unfold candidates
  have row : edge 2753 = SearchData.row2753.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan326_1 : scan edge (search edge 3) 4 [2207] 2207 innerTail326_1 best = best := by
  rw [innerTail326_1, scan, if_neg (by decide), innerCandidates326_1_checked]
  change scan edge (search edge 3) 4 [2207] 2207 innerTail326_2
    (if 3 ≤ innerCandidates326_1.length then search edge 3 [2207, 2753] innerCandidates326_1 4960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan326_2

def innerTail326_0 : List Nat := 2339 :: innerTail326_1
def innerCandidates326_0 : List Nat := [3347, 4337]
set_option maxRecDepth 4096 in
theorem innerCandidates326_0_checked : candidates edge 4 2207 2339 innerTail326_1 best.bound = innerCandidates326_0 := by
  unfold candidates
  have row : edge 2339 = PrimePairCache.smallRow2339.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan326_0 : scan edge (search edge 3) 4 [2207] 2207 innerTail326_0 best = best := by
  rw [innerTail326_0, scan, if_neg (by decide), innerCandidates326_0_checked]
  change scan edge (search edge 3) 4 [2207] 2207 innerTail326_1
    (if 3 ≤ innerCandidates326_0.length then search edge 3 [2207, 2339] innerCandidates326_0 4546 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan326_1

theorem rootBranch326 : search edge 4 [2207] rootCandidates326 2207 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2207] 2207 innerTail326_0 best = best
  exact innerScan326_0

def rootCandidates325 : List Nat := [2371, 2887, 3061, 5521, 5557, 5623, 5647, 5857, 6037, 6271, 6529, 6961, 7789, 7927, 8161]
set_option maxRecDepth 4096 in
theorem rootCandidates325_checked : candidates edge 5 0 2203 tail326 best.bound = rootCandidates325 := by
  unfold candidates
  have row : edge 2203 = PrimePairCache.smallRow2203.contains := by funext q; rfl
  rw [row]
  decide

def innerTail325_15 : List Nat := []
theorem innerScan325_15 : scan edge (search edge 3) 4 [2203] 2203 innerTail325_15 best = best := rfl

def innerTail325_14 : List Nat := 8161 :: innerTail325_15
def innerCandidates325_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates325_14_checked : candidates edge 4 2203 8161 innerTail325_15 best.bound = innerCandidates325_14 := by
  unfold candidates
  have row : edge 8161 = SearchData.row8161.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan325_14 : scan edge (search edge 3) 4 [2203] 2203 innerTail325_14 best = best := by
  rw [innerTail325_14, scan, if_neg (by decide), innerCandidates325_14_checked]
  change scan edge (search edge 3) 4 [2203] 2203 innerTail325_15
    (if 3 ≤ innerCandidates325_14.length then search edge 3 [2203, 8161] innerCandidates325_14 10364 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan325_15

def innerTail325_13 : List Nat := 7927 :: innerTail325_14
def innerCandidates325_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates325_13_checked : candidates edge 4 2203 7927 innerTail325_14 best.bound = innerCandidates325_13 := by
  unfold candidates
  have row : edge 7927 = SearchData.row7927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan325_13 : scan edge (search edge 3) 4 [2203] 2203 innerTail325_13 best = best := by
  rw [innerTail325_13, scan, if_neg (by decide), innerCandidates325_13_checked]
  change scan edge (search edge 3) 4 [2203] 2203 innerTail325_14
    (if 3 ≤ innerCandidates325_13.length then search edge 3 [2203, 7927] innerCandidates325_13 10130 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan325_14

def innerTail325_12 : List Nat := 7789 :: innerTail325_13
def innerCandidates325_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates325_12_checked : candidates edge 4 2203 7789 innerTail325_13 best.bound = innerCandidates325_12 := by
  unfold candidates
  have row : edge 7789 = SearchData.row7789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan325_12 : scan edge (search edge 3) 4 [2203] 2203 innerTail325_12 best = best := by
  rw [innerTail325_12, scan, if_neg (by decide), innerCandidates325_12_checked]
  change scan edge (search edge 3) 4 [2203] 2203 innerTail325_13
    (if 3 ≤ innerCandidates325_12.length then search edge 3 [2203, 7789] innerCandidates325_12 9992 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan325_13

def innerTail325_11 : List Nat := 6961 :: innerTail325_12
def innerCandidates325_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates325_11_checked : candidates edge 4 2203 6961 innerTail325_12 best.bound = innerCandidates325_11 := by
  unfold candidates
  have row : edge 6961 = SearchData.row6961.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan325_11 : scan edge (search edge 3) 4 [2203] 2203 innerTail325_11 best = best := by
  rw [innerTail325_11, scan, if_neg (by decide), innerCandidates325_11_checked]
  change scan edge (search edge 3) 4 [2203] 2203 innerTail325_12
    (if 3 ≤ innerCandidates325_11.length then search edge 3 [2203, 6961] innerCandidates325_11 9164 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan325_12

def innerTail325_10 : List Nat := 6529 :: innerTail325_11
def innerCandidates325_10 : List Nat := [6961]
set_option maxRecDepth 4096 in
theorem innerCandidates325_10_checked : candidates edge 4 2203 6529 innerTail325_11 best.bound = innerCandidates325_10 := by
  unfold candidates
  have row : edge 6529 = SearchData.row6529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan325_10 : scan edge (search edge 3) 4 [2203] 2203 innerTail325_10 best = best := by
  rw [innerTail325_10, scan, if_neg (by decide), innerCandidates325_10_checked]
  change scan edge (search edge 3) 4 [2203] 2203 innerTail325_11
    (if 3 ≤ innerCandidates325_10.length then search edge 3 [2203, 6529] innerCandidates325_10 8732 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan325_11

def innerTail325_9 : List Nat := 6271 :: innerTail325_10
def innerCandidates325_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates325_9_checked : candidates edge 4 2203 6271 innerTail325_10 best.bound = innerCandidates325_9 := by
  unfold candidates
  have row : edge 6271 = SearchData.row6271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan325_9 : scan edge (search edge 3) 4 [2203] 2203 innerTail325_9 best = best := by
  rw [innerTail325_9, scan, if_neg (by decide), innerCandidates325_9_checked]
  change scan edge (search edge 3) 4 [2203] 2203 innerTail325_10
    (if 3 ≤ innerCandidates325_9.length then search edge 3 [2203, 6271] innerCandidates325_9 8474 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan325_10

def innerTail325_8 : List Nat := 6037 :: innerTail325_9
def innerCandidates325_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates325_8_checked : candidates edge 4 2203 6037 innerTail325_9 best.bound = innerCandidates325_8 := by
  unfold candidates
  have row : edge 6037 = SearchData.row6037.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan325_8 : scan edge (search edge 3) 4 [2203] 2203 innerTail325_8 best = best := by
  rw [innerTail325_8, scan, if_neg (by decide), innerCandidates325_8_checked]
  change scan edge (search edge 3) 4 [2203] 2203 innerTail325_9
    (if 3 ≤ innerCandidates325_8.length then search edge 3 [2203, 6037] innerCandidates325_8 8240 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan325_9

def innerTail325_7 : List Nat := 5857 :: innerTail325_8
def innerCandidates325_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates325_7_checked : candidates edge 4 2203 5857 innerTail325_8 best.bound = innerCandidates325_7 := by
  unfold candidates
  have row : edge 5857 = SearchData.row5857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan325_7 : scan edge (search edge 3) 4 [2203] 2203 innerTail325_7 best = best := by
  rw [innerTail325_7, scan, if_neg (by decide), innerCandidates325_7_checked]
  change scan edge (search edge 3) 4 [2203] 2203 innerTail325_8
    (if 3 ≤ innerCandidates325_7.length then search edge 3 [2203, 5857] innerCandidates325_7 8060 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan325_8

def innerTail325_6 : List Nat := 5647 :: innerTail325_7
def innerCandidates325_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates325_6_checked : candidates edge 4 2203 5647 innerTail325_7 best.bound = innerCandidates325_6 := by
  unfold candidates
  have row : edge 5647 = SearchData.row5647.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan325_6 : scan edge (search edge 3) 4 [2203] 2203 innerTail325_6 best = best := by
  rw [innerTail325_6, scan, if_neg (by decide), innerCandidates325_6_checked]
  change scan edge (search edge 3) 4 [2203] 2203 innerTail325_7
    (if 3 ≤ innerCandidates325_6.length then search edge 3 [2203, 5647] innerCandidates325_6 7850 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan325_7

def innerTail325_5 : List Nat := 5623 :: innerTail325_6
def innerCandidates325_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates325_5_checked : candidates edge 4 2203 5623 innerTail325_6 best.bound = innerCandidates325_5 := by
  unfold candidates
  have row : edge 5623 = SearchData.row5623.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan325_5 : scan edge (search edge 3) 4 [2203] 2203 innerTail325_5 best = best := by
  rw [innerTail325_5, scan, if_neg (by decide), innerCandidates325_5_checked]
  change scan edge (search edge 3) 4 [2203] 2203 innerTail325_6
    (if 3 ≤ innerCandidates325_5.length then search edge 3 [2203, 5623] innerCandidates325_5 7826 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan325_6

def innerTail325_4 : List Nat := 5557 :: innerTail325_5
def innerCandidates325_4 : List Nat := [6529, 7789]
set_option maxRecDepth 4096 in
theorem innerCandidates325_4_checked : candidates edge 4 2203 5557 innerTail325_5 best.bound = innerCandidates325_4 := by
  unfold candidates
  have row : edge 5557 = SearchData.row5557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan325_4 : scan edge (search edge 3) 4 [2203] 2203 innerTail325_4 best = best := by
  rw [innerTail325_4, scan, if_neg (by decide), innerCandidates325_4_checked]
  change scan edge (search edge 3) 4 [2203] 2203 innerTail325_5
    (if 3 ≤ innerCandidates325_4.length then search edge 3 [2203, 5557] innerCandidates325_4 7760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan325_5

def innerTail325_3 : List Nat := 5521 :: innerTail325_4
def innerCandidates325_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates325_3_checked : candidates edge 4 2203 5521 innerTail325_4 best.bound = innerCandidates325_3 := by
  unfold candidates
  have row : edge 5521 = SearchData.row5521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan325_3 : scan edge (search edge 3) 4 [2203] 2203 innerTail325_3 best = best := by
  rw [innerTail325_3, scan, if_neg (by decide), innerCandidates325_3_checked]
  change scan edge (search edge 3) 4 [2203] 2203 innerTail325_4
    (if 3 ≤ innerCandidates325_3.length then search edge 3 [2203, 5521] innerCandidates325_3 7724 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan325_4

def innerTail325_2 : List Nat := 3061 :: innerTail325_3
def innerCandidates325_2 : List Nat := [5857]
set_option maxRecDepth 4096 in
theorem innerCandidates325_2_checked : candidates edge 4 2203 3061 innerTail325_3 best.bound = innerCandidates325_2 := by
  unfold candidates
  have row : edge 3061 = SearchData.row3061.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan325_2 : scan edge (search edge 3) 4 [2203] 2203 innerTail325_2 best = best := by
  rw [innerTail325_2, scan, if_neg (by decide), innerCandidates325_2_checked]
  change scan edge (search edge 3) 4 [2203] 2203 innerTail325_3
    (if 3 ≤ innerCandidates325_2.length then search edge 3 [2203, 3061] innerCandidates325_2 5264 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan325_3

def innerTail325_1 : List Nat := 2887 :: innerTail325_2
def innerCandidates325_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates325_1_checked : candidates edge 4 2203 2887 innerTail325_2 best.bound = innerCandidates325_1 := by
  unfold candidates
  have row : edge 2887 = SearchData.row2887.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan325_1 : scan edge (search edge 3) 4 [2203] 2203 innerTail325_1 best = best := by
  rw [innerTail325_1, scan, if_neg (by decide), innerCandidates325_1_checked]
  change scan edge (search edge 3) 4 [2203] 2203 innerTail325_2
    (if 3 ≤ innerCandidates325_1.length then search edge 3 [2203, 2887] innerCandidates325_1 5090 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan325_2

def innerTail325_0 : List Nat := 2371 :: innerTail325_1
def innerCandidates325_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates325_0_checked : candidates edge 4 2203 2371 innerTail325_1 best.bound = innerCandidates325_0 := by
  unfold candidates
  have row : edge 2371 = PrimePairCache.smallRow2371.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan325_0 : scan edge (search edge 3) 4 [2203] 2203 innerTail325_0 best = best := by
  rw [innerTail325_0, scan, if_neg (by decide), innerCandidates325_0_checked]
  change scan edge (search edge 3) 4 [2203] 2203 innerTail325_1
    (if 3 ≤ innerCandidates325_0.length then search edge 3 [2203, 2371] innerCandidates325_0 4574 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan325_1

theorem rootBranch325 : search edge 4 [2203] rootCandidates325 2203 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2203] 2203 innerTail325_0 best = best
  exact innerScan325_0

end ProjectEulerSolutions.PrimePairUpper.Search
