import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates295 : List Nat := [1979, 2381, 2591, 4397, 4409, 4457, 4493, 4751, 4919, 5711, 5741, 5897, 6131, 7547, 8291, 8363]
set_option maxRecDepth 4096 in
theorem rootCandidates295_checked : candidates edge 5 0 1973 tail296 best.bound = rootCandidates295 := by
  unfold candidates
  have row : edge 1973 = PrimePairCache.smallRow1973.contains := by funext q; rfl
  rw [row]
  decide

def innerTail295_16 : List Nat := []
theorem innerScan295_16 : scan edge (search edge 3) 4 [1973] 1973 innerTail295_16 best = best := rfl

def innerTail295_15 : List Nat := 8363 :: innerTail295_16
def innerCandidates295_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates295_15_checked : candidates edge 4 1973 8363 innerTail295_16 best.bound = innerCandidates295_15 := by
  unfold candidates
  have row : edge 8363 = SearchData.row8363.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan295_15 : scan edge (search edge 3) 4 [1973] 1973 innerTail295_15 best = best := by
  rw [innerTail295_15, scan, if_neg (by decide), innerCandidates295_15_checked]
  change scan edge (search edge 3) 4 [1973] 1973 innerTail295_16
    (if 3 ≤ innerCandidates295_15.length then search edge 3 [1973, 8363] innerCandidates295_15 10336 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan295_16

def innerTail295_14 : List Nat := 8291 :: innerTail295_15
def innerCandidates295_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates295_14_checked : candidates edge 4 1973 8291 innerTail295_15 best.bound = innerCandidates295_14 := by
  unfold candidates
  have row : edge 8291 = SearchData.row8291.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan295_14 : scan edge (search edge 3) 4 [1973] 1973 innerTail295_14 best = best := by
  rw [innerTail295_14, scan, if_neg (by decide), innerCandidates295_14_checked]
  change scan edge (search edge 3) 4 [1973] 1973 innerTail295_15
    (if 3 ≤ innerCandidates295_14.length then search edge 3 [1973, 8291] innerCandidates295_14 10264 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan295_15

def innerTail295_13 : List Nat := 7547 :: innerTail295_14
def innerCandidates295_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates295_13_checked : candidates edge 4 1973 7547 innerTail295_14 best.bound = innerCandidates295_13 := by
  unfold candidates
  have row : edge 7547 = SearchData.row7547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan295_13 : scan edge (search edge 3) 4 [1973] 1973 innerTail295_13 best = best := by
  rw [innerTail295_13, scan, if_neg (by decide), innerCandidates295_13_checked]
  change scan edge (search edge 3) 4 [1973] 1973 innerTail295_14
    (if 3 ≤ innerCandidates295_13.length then search edge 3 [1973, 7547] innerCandidates295_13 9520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan295_14

def innerTail295_12 : List Nat := 6131 :: innerTail295_13
def innerCandidates295_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates295_12_checked : candidates edge 4 1973 6131 innerTail295_13 best.bound = innerCandidates295_12 := by
  unfold candidates
  have row : edge 6131 = SearchData.row6131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan295_12 : scan edge (search edge 3) 4 [1973] 1973 innerTail295_12 best = best := by
  rw [innerTail295_12, scan, if_neg (by decide), innerCandidates295_12_checked]
  change scan edge (search edge 3) 4 [1973] 1973 innerTail295_13
    (if 3 ≤ innerCandidates295_12.length then search edge 3 [1973, 6131] innerCandidates295_12 8104 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan295_13

def innerTail295_11 : List Nat := 5897 :: innerTail295_12
def innerCandidates295_11 : List Nat := [6131]
set_option maxRecDepth 4096 in
theorem innerCandidates295_11_checked : candidates edge 4 1973 5897 innerTail295_12 best.bound = innerCandidates295_11 := by
  unfold candidates
  have row : edge 5897 = SearchData.row5897.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan295_11 : scan edge (search edge 3) 4 [1973] 1973 innerTail295_11 best = best := by
  rw [innerTail295_11, scan, if_neg (by decide), innerCandidates295_11_checked]
  change scan edge (search edge 3) 4 [1973] 1973 innerTail295_12
    (if 3 ≤ innerCandidates295_11.length then search edge 3 [1973, 5897] innerCandidates295_11 7870 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan295_12

def innerTail295_10 : List Nat := 5741 :: innerTail295_11
def innerCandidates295_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates295_10_checked : candidates edge 4 1973 5741 innerTail295_11 best.bound = innerCandidates295_10 := by
  unfold candidates
  have row : edge 5741 = SearchData.row5741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan295_10 : scan edge (search edge 3) 4 [1973] 1973 innerTail295_10 best = best := by
  rw [innerTail295_10, scan, if_neg (by decide), innerCandidates295_10_checked]
  change scan edge (search edge 3) 4 [1973] 1973 innerTail295_11
    (if 3 ≤ innerCandidates295_10.length then search edge 3 [1973, 5741] innerCandidates295_10 7714 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan295_11

def innerTail295_9 : List Nat := 5711 :: innerTail295_10
def innerCandidates295_9 : List Nat := [8291]
set_option maxRecDepth 4096 in
theorem innerCandidates295_9_checked : candidates edge 4 1973 5711 innerTail295_10 best.bound = innerCandidates295_9 := by
  unfold candidates
  have row : edge 5711 = SearchData.row5711.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan295_9 : scan edge (search edge 3) 4 [1973] 1973 innerTail295_9 best = best := by
  rw [innerTail295_9, scan, if_neg (by decide), innerCandidates295_9_checked]
  change scan edge (search edge 3) 4 [1973] 1973 innerTail295_10
    (if 3 ≤ innerCandidates295_9.length then search edge 3 [1973, 5711] innerCandidates295_9 7684 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan295_10

def innerTail295_8 : List Nat := 4919 :: innerTail295_9
def innerCandidates295_8 : List Nat := [5897]
set_option maxRecDepth 4096 in
theorem innerCandidates295_8_checked : candidates edge 4 1973 4919 innerTail295_9 best.bound = innerCandidates295_8 := by
  unfold candidates
  have row : edge 4919 = SearchData.row4919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan295_8 : scan edge (search edge 3) 4 [1973] 1973 innerTail295_8 best = best := by
  rw [innerTail295_8, scan, if_neg (by decide), innerCandidates295_8_checked]
  change scan edge (search edge 3) 4 [1973] 1973 innerTail295_9
    (if 3 ≤ innerCandidates295_8.length then search edge 3 [1973, 4919] innerCandidates295_8 6892 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan295_9

def innerTail295_7 : List Nat := 4751 :: innerTail295_8
def innerCandidates295_7 : List Nat := [5711]
set_option maxRecDepth 4096 in
theorem innerCandidates295_7_checked : candidates edge 4 1973 4751 innerTail295_8 best.bound = innerCandidates295_7 := by
  unfold candidates
  have row : edge 4751 = SearchData.row4751.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan295_7 : scan edge (search edge 3) 4 [1973] 1973 innerTail295_7 best = best := by
  rw [innerTail295_7, scan, if_neg (by decide), innerCandidates295_7_checked]
  change scan edge (search edge 3) 4 [1973] 1973 innerTail295_8
    (if 3 ≤ innerCandidates295_7.length then search edge 3 [1973, 4751] innerCandidates295_7 6724 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan295_8

def innerTail295_6 : List Nat := 4493 :: innerTail295_7
def innerCandidates295_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates295_6_checked : candidates edge 4 1973 4493 innerTail295_7 best.bound = innerCandidates295_6 := by
  unfold candidates
  have row : edge 4493 = SearchData.row4493.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan295_6 : scan edge (search edge 3) 4 [1973] 1973 innerTail295_6 best = best := by
  rw [innerTail295_6, scan, if_neg (by decide), innerCandidates295_6_checked]
  change scan edge (search edge 3) 4 [1973] 1973 innerTail295_7
    (if 3 ≤ innerCandidates295_6.length then search edge 3 [1973, 4493] innerCandidates295_6 6466 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan295_7

def innerTail295_5 : List Nat := 4457 :: innerTail295_6
def innerCandidates295_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates295_5_checked : candidates edge 4 1973 4457 innerTail295_6 best.bound = innerCandidates295_5 := by
  unfold candidates
  have row : edge 4457 = SearchData.row4457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan295_5 : scan edge (search edge 3) 4 [1973] 1973 innerTail295_5 best = best := by
  rw [innerTail295_5, scan, if_neg (by decide), innerCandidates295_5_checked]
  change scan edge (search edge 3) 4 [1973] 1973 innerTail295_6
    (if 3 ≤ innerCandidates295_5.length then search edge 3 [1973, 4457] innerCandidates295_5 6430 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan295_6

def innerTail295_4 : List Nat := 4409 :: innerTail295_5
def innerCandidates295_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates295_4_checked : candidates edge 4 1973 4409 innerTail295_5 best.bound = innerCandidates295_4 := by
  unfold candidates
  have row : edge 4409 = SearchData.row4409.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan295_4 : scan edge (search edge 3) 4 [1973] 1973 innerTail295_4 best = best := by
  rw [innerTail295_4, scan, if_neg (by decide), innerCandidates295_4_checked]
  change scan edge (search edge 3) 4 [1973] 1973 innerTail295_5
    (if 3 ≤ innerCandidates295_4.length then search edge 3 [1973, 4409] innerCandidates295_4 6382 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan295_5

def innerTail295_3 : List Nat := 4397 :: innerTail295_4
def innerCandidates295_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates295_3_checked : candidates edge 4 1973 4397 innerTail295_4 best.bound = innerCandidates295_3 := by
  unfold candidates
  have row : edge 4397 = SearchData.row4397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan295_3 : scan edge (search edge 3) 4 [1973] 1973 innerTail295_3 best = best := by
  rw [innerTail295_3, scan, if_neg (by decide), innerCandidates295_3_checked]
  change scan edge (search edge 3) 4 [1973] 1973 innerTail295_4
    (if 3 ≤ innerCandidates295_3.length then search edge 3 [1973, 4397] innerCandidates295_3 6370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan295_4

def innerTail295_2 : List Nat := 2591 :: innerTail295_3
def innerCandidates295_2 : List Nat := [5741]
set_option maxRecDepth 4096 in
theorem innerCandidates295_2_checked : candidates edge 4 1973 2591 innerTail295_3 best.bound = innerCandidates295_2 := by
  unfold candidates
  have row : edge 2591 = PrimePairCache.smallRow2591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan295_2 : scan edge (search edge 3) 4 [1973] 1973 innerTail295_2 best = best := by
  rw [innerTail295_2, scan, if_neg (by decide), innerCandidates295_2_checked]
  change scan edge (search edge 3) 4 [1973] 1973 innerTail295_3
    (if 3 ≤ innerCandidates295_2.length then search edge 3 [1973, 2591] innerCandidates295_2 4564 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan295_3

def innerTail295_1 : List Nat := 2381 :: innerTail295_2
def innerCandidates295_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates295_1_checked : candidates edge 4 1973 2381 innerTail295_2 best.bound = innerCandidates295_1 := by
  unfold candidates
  have row : edge 2381 = PrimePairCache.smallRow2381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan295_1 : scan edge (search edge 3) 4 [1973] 1973 innerTail295_1 best = best := by
  rw [innerTail295_1, scan, if_neg (by decide), innerCandidates295_1_checked]
  change scan edge (search edge 3) 4 [1973] 1973 innerTail295_2
    (if 3 ≤ innerCandidates295_1.length then search edge 3 [1973, 2381] innerCandidates295_1 4354 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan295_2

def innerTail295_0 : List Nat := 1979 :: innerTail295_1
def innerCandidates295_0 : List Nat := [8291]
set_option maxRecDepth 4096 in
theorem innerCandidates295_0_checked : candidates edge 4 1973 1979 innerTail295_1 best.bound = innerCandidates295_0 := by
  unfold candidates
  have row : edge 1979 = PrimePairCache.smallRow1979.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan295_0 : scan edge (search edge 3) 4 [1973] 1973 innerTail295_0 best = best := by
  rw [innerTail295_0, scan, if_neg (by decide), innerCandidates295_0_checked]
  change scan edge (search edge 3) 4 [1973] 1973 innerTail295_1
    (if 3 ≤ innerCandidates295_0.length then search edge 3 [1973, 1979] innerCandidates295_0 3952 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan295_1

theorem rootBranch295 : search edge 4 [1973] rootCandidates295 1973 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1973] 1973 innerTail295_0 best = best
  exact innerScan295_0

def rootCandidates294 : List Nat := [2341, 2683, 3547, 3643, 3733, 4549, 4783, 4909, 5227, 5437, 6037, 6151, 6379, 6547, 6823, 7027, 7489, 8167]
set_option maxRecDepth 4096 in
theorem rootCandidates294_checked : candidates edge 5 0 1951 tail295 best.bound = rootCandidates294 := by
  unfold candidates
  have row : edge 1951 = PrimePairCache.smallRow1951.contains := by funext q; rfl
  rw [row]
  decide

def innerTail294_18 : List Nat := []
theorem innerScan294_18 : scan edge (search edge 3) 4 [1951] 1951 innerTail294_18 best = best := rfl

def innerTail294_17 : List Nat := 8167 :: innerTail294_18
def innerCandidates294_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates294_17_checked : candidates edge 4 1951 8167 innerTail294_18 best.bound = innerCandidates294_17 := by
  unfold candidates
  have row : edge 8167 = SearchData.row8167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan294_17 : scan edge (search edge 3) 4 [1951] 1951 innerTail294_17 best = best := by
  rw [innerTail294_17, scan, if_neg (by decide), innerCandidates294_17_checked]
  change scan edge (search edge 3) 4 [1951] 1951 innerTail294_18
    (if 3 ≤ innerCandidates294_17.length then search edge 3 [1951, 8167] innerCandidates294_17 10118 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan294_18

def innerTail294_16 : List Nat := 7489 :: innerTail294_17
def innerCandidates294_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates294_16_checked : candidates edge 4 1951 7489 innerTail294_17 best.bound = innerCandidates294_16 := by
  unfold candidates
  have row : edge 7489 = SearchData.row7489.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan294_16 : scan edge (search edge 3) 4 [1951] 1951 innerTail294_16 best = best := by
  rw [innerTail294_16, scan, if_neg (by decide), innerCandidates294_16_checked]
  change scan edge (search edge 3) 4 [1951] 1951 innerTail294_17
    (if 3 ≤ innerCandidates294_16.length then search edge 3 [1951, 7489] innerCandidates294_16 9440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan294_17

def innerTail294_15 : List Nat := 7027 :: innerTail294_16
def innerCandidates294_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates294_15_checked : candidates edge 4 1951 7027 innerTail294_16 best.bound = innerCandidates294_15 := by
  unfold candidates
  have row : edge 7027 = SearchData.row7027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan294_15 : scan edge (search edge 3) 4 [1951] 1951 innerTail294_15 best = best := by
  rw [innerTail294_15, scan, if_neg (by decide), innerCandidates294_15_checked]
  change scan edge (search edge 3) 4 [1951] 1951 innerTail294_16
    (if 3 ≤ innerCandidates294_15.length then search edge 3 [1951, 7027] innerCandidates294_15 8978 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan294_16

def innerTail294_14 : List Nat := 6823 :: innerTail294_15
def innerCandidates294_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates294_14_checked : candidates edge 4 1951 6823 innerTail294_15 best.bound = innerCandidates294_14 := by
  unfold candidates
  have row : edge 6823 = SearchData.row6823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan294_14 : scan edge (search edge 3) 4 [1951] 1951 innerTail294_14 best = best := by
  rw [innerTail294_14, scan, if_neg (by decide), innerCandidates294_14_checked]
  change scan edge (search edge 3) 4 [1951] 1951 innerTail294_15
    (if 3 ≤ innerCandidates294_14.length then search edge 3 [1951, 6823] innerCandidates294_14 8774 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan294_15

def innerTail294_13 : List Nat := 6547 :: innerTail294_14
def innerCandidates294_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates294_13_checked : candidates edge 4 1951 6547 innerTail294_14 best.bound = innerCandidates294_13 := by
  unfold candidates
  have row : edge 6547 = SearchData.row6547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan294_13 : scan edge (search edge 3) 4 [1951] 1951 innerTail294_13 best = best := by
  rw [innerTail294_13, scan, if_neg (by decide), innerCandidates294_13_checked]
  change scan edge (search edge 3) 4 [1951] 1951 innerTail294_14
    (if 3 ≤ innerCandidates294_13.length then search edge 3 [1951, 6547] innerCandidates294_13 8498 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan294_14

def innerTail294_12 : List Nat := 6379 :: innerTail294_13
def innerCandidates294_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates294_12_checked : candidates edge 4 1951 6379 innerTail294_13 best.bound = innerCandidates294_12 := by
  unfold candidates
  have row : edge 6379 = SearchData.row6379.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan294_12 : scan edge (search edge 3) 4 [1951] 1951 innerTail294_12 best = best := by
  rw [innerTail294_12, scan, if_neg (by decide), innerCandidates294_12_checked]
  change scan edge (search edge 3) 4 [1951] 1951 innerTail294_13
    (if 3 ≤ innerCandidates294_12.length then search edge 3 [1951, 6379] innerCandidates294_12 8330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan294_13

def innerTail294_11 : List Nat := 6151 :: innerTail294_12
def innerCandidates294_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates294_11_checked : candidates edge 4 1951 6151 innerTail294_12 best.bound = innerCandidates294_11 := by
  unfold candidates
  have row : edge 6151 = SearchData.row6151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan294_11 : scan edge (search edge 3) 4 [1951] 1951 innerTail294_11 best = best := by
  rw [innerTail294_11, scan, if_neg (by decide), innerCandidates294_11_checked]
  change scan edge (search edge 3) 4 [1951] 1951 innerTail294_12
    (if 3 ≤ innerCandidates294_11.length then search edge 3 [1951, 6151] innerCandidates294_11 8102 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan294_12

def innerTail294_10 : List Nat := 6037 :: innerTail294_11
def innerCandidates294_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates294_10_checked : candidates edge 4 1951 6037 innerTail294_11 best.bound = innerCandidates294_10 := by
  unfold candidates
  have row : edge 6037 = SearchData.row6037.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan294_10 : scan edge (search edge 3) 4 [1951] 1951 innerTail294_10 best = best := by
  rw [innerTail294_10, scan, if_neg (by decide), innerCandidates294_10_checked]
  change scan edge (search edge 3) 4 [1951] 1951 innerTail294_11
    (if 3 ≤ innerCandidates294_10.length then search edge 3 [1951, 6037] innerCandidates294_10 7988 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan294_11

def innerTail294_9 : List Nat := 5437 :: innerTail294_10
def innerCandidates294_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates294_9_checked : candidates edge 4 1951 5437 innerTail294_10 best.bound = innerCandidates294_9 := by
  unfold candidates
  have row : edge 5437 = SearchData.row5437.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan294_9 : scan edge (search edge 3) 4 [1951] 1951 innerTail294_9 best = best := by
  rw [innerTail294_9, scan, if_neg (by decide), innerCandidates294_9_checked]
  change scan edge (search edge 3) 4 [1951] 1951 innerTail294_10
    (if 3 ≤ innerCandidates294_9.length then search edge 3 [1951, 5437] innerCandidates294_9 7388 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan294_10

def innerTail294_8 : List Nat := 5227 :: innerTail294_9
def innerCandidates294_8 : List Nat := [6379]
set_option maxRecDepth 4096 in
theorem innerCandidates294_8_checked : candidates edge 4 1951 5227 innerTail294_9 best.bound = innerCandidates294_8 := by
  unfold candidates
  have row : edge 5227 = SearchData.row5227.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan294_8 : scan edge (search edge 3) 4 [1951] 1951 innerTail294_8 best = best := by
  rw [innerTail294_8, scan, if_neg (by decide), innerCandidates294_8_checked]
  change scan edge (search edge 3) 4 [1951] 1951 innerTail294_9
    (if 3 ≤ innerCandidates294_8.length then search edge 3 [1951, 5227] innerCandidates294_8 7178 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan294_9

def innerTail294_7 : List Nat := 4909 :: innerTail294_8
def innerCandidates294_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates294_7_checked : candidates edge 4 1951 4909 innerTail294_8 best.bound = innerCandidates294_7 := by
  unfold candidates
  have row : edge 4909 = SearchData.row4909.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan294_7 : scan edge (search edge 3) 4 [1951] 1951 innerTail294_7 best = best := by
  rw [innerTail294_7, scan, if_neg (by decide), innerCandidates294_7_checked]
  change scan edge (search edge 3) 4 [1951] 1951 innerTail294_8
    (if 3 ≤ innerCandidates294_7.length then search edge 3 [1951, 4909] innerCandidates294_7 6860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan294_8

def innerTail294_6 : List Nat := 4783 :: innerTail294_7
def innerCandidates294_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates294_6_checked : candidates edge 4 1951 4783 innerTail294_7 best.bound = innerCandidates294_6 := by
  unfold candidates
  have row : edge 4783 = SearchData.row4783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan294_6 : scan edge (search edge 3) 4 [1951] 1951 innerTail294_6 best = best := by
  rw [innerTail294_6, scan, if_neg (by decide), innerCandidates294_6_checked]
  change scan edge (search edge 3) 4 [1951] 1951 innerTail294_7
    (if 3 ≤ innerCandidates294_6.length then search edge 3 [1951, 4783] innerCandidates294_6 6734 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan294_7

def innerTail294_5 : List Nat := 4549 :: innerTail294_6
def innerCandidates294_5 : List Nat := [6151]
set_option maxRecDepth 4096 in
theorem innerCandidates294_5_checked : candidates edge 4 1951 4549 innerTail294_6 best.bound = innerCandidates294_5 := by
  unfold candidates
  have row : edge 4549 = SearchData.row4549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan294_5 : scan edge (search edge 3) 4 [1951] 1951 innerTail294_5 best = best := by
  rw [innerTail294_5, scan, if_neg (by decide), innerCandidates294_5_checked]
  change scan edge (search edge 3) 4 [1951] 1951 innerTail294_6
    (if 3 ≤ innerCandidates294_5.length then search edge 3 [1951, 4549] innerCandidates294_5 6500 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan294_6

def innerTail294_4 : List Nat := 3733 :: innerTail294_5
def innerCandidates294_4 : List Nat := [6151]
set_option maxRecDepth 4096 in
theorem innerCandidates294_4_checked : candidates edge 4 1951 3733 innerTail294_5 best.bound = innerCandidates294_4 := by
  unfold candidates
  have row : edge 3733 = SearchData.row3733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan294_4 : scan edge (search edge 3) 4 [1951] 1951 innerTail294_4 best = best := by
  rw [innerTail294_4, scan, if_neg (by decide), innerCandidates294_4_checked]
  change scan edge (search edge 3) 4 [1951] 1951 innerTail294_5
    (if 3 ≤ innerCandidates294_4.length then search edge 3 [1951, 3733] innerCandidates294_4 5684 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan294_5

def innerTail294_3 : List Nat := 3643 :: innerTail294_4
def innerCandidates294_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates294_3_checked : candidates edge 4 1951 3643 innerTail294_4 best.bound = innerCandidates294_3 := by
  unfold candidates
  have row : edge 3643 = SearchData.row3643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan294_3 : scan edge (search edge 3) 4 [1951] 1951 innerTail294_3 best = best := by
  rw [innerTail294_3, scan, if_neg (by decide), innerCandidates294_3_checked]
  change scan edge (search edge 3) 4 [1951] 1951 innerTail294_4
    (if 3 ≤ innerCandidates294_3.length then search edge 3 [1951, 3643] innerCandidates294_3 5594 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan294_4

def innerTail294_2 : List Nat := 3547 :: innerTail294_3
def innerCandidates294_2 : List Nat := [3643, 6151]
set_option maxRecDepth 4096 in
theorem innerCandidates294_2_checked : candidates edge 4 1951 3547 innerTail294_3 best.bound = innerCandidates294_2 := by
  unfold candidates
  have row : edge 3547 = SearchData.row3547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan294_2 : scan edge (search edge 3) 4 [1951] 1951 innerTail294_2 best = best := by
  rw [innerTail294_2, scan, if_neg (by decide), innerCandidates294_2_checked]
  change scan edge (search edge 3) 4 [1951] 1951 innerTail294_3
    (if 3 ≤ innerCandidates294_2.length then search edge 3 [1951, 3547] innerCandidates294_2 5498 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan294_3

def innerTail294_1 : List Nat := 2683 :: innerTail294_2
def innerCandidates294_1 : List Nat := [8167]
set_option maxRecDepth 4096 in
theorem innerCandidates294_1_checked : candidates edge 4 1951 2683 innerTail294_2 best.bound = innerCandidates294_1 := by
  unfold candidates
  have row : edge 2683 = SearchData.row2683.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan294_1 : scan edge (search edge 3) 4 [1951] 1951 innerTail294_1 best = best := by
  rw [innerTail294_1, scan, if_neg (by decide), innerCandidates294_1_checked]
  change scan edge (search edge 3) 4 [1951] 1951 innerTail294_2
    (if 3 ≤ innerCandidates294_1.length then search edge 3 [1951, 2683] innerCandidates294_1 4634 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan294_2

def innerTail294_0 : List Nat := 2341 :: innerTail294_1
def innerCandidates294_0 : List Nat := [7489]
set_option maxRecDepth 4096 in
theorem innerCandidates294_0_checked : candidates edge 4 1951 2341 innerTail294_1 best.bound = innerCandidates294_0 := by
  unfold candidates
  have row : edge 2341 = PrimePairCache.smallRow2341.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan294_0 : scan edge (search edge 3) 4 [1951] 1951 innerTail294_0 best = best := by
  rw [innerTail294_0, scan, if_neg (by decide), innerCandidates294_0_checked]
  change scan edge (search edge 3) 4 [1951] 1951 innerTail294_1
    (if 3 ≤ innerCandidates294_0.length then search edge 3 [1951, 2341] innerCandidates294_0 4292 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan294_1

theorem rootBranch294 : search edge 4 [1951] rootCandidates294 1951 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1951] 1951 innerTail294_0 best = best
  exact innerScan294_0

def rootCandidates293 : List Nat := [2027, 2621, 2657, 2663, 2789, 2819, 2903, 3911, 3923, 3989, 4337, 5669, 6863, 7727, 7901]
set_option maxRecDepth 4096 in
theorem rootCandidates293_checked : candidates edge 5 0 1949 tail294 best.bound = rootCandidates293 := by
  unfold candidates
  have row : edge 1949 = PrimePairCache.smallRow1949.contains := by funext q; rfl
  rw [row]
  decide

def innerTail293_15 : List Nat := []
theorem innerScan293_15 : scan edge (search edge 3) 4 [1949] 1949 innerTail293_15 best = best := rfl

def innerTail293_14 : List Nat := 7901 :: innerTail293_15
def innerCandidates293_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates293_14_checked : candidates edge 4 1949 7901 innerTail293_15 best.bound = innerCandidates293_14 := by
  unfold candidates
  have row : edge 7901 = SearchData.row7901.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan293_14 : scan edge (search edge 3) 4 [1949] 1949 innerTail293_14 best = best := by
  rw [innerTail293_14, scan, if_neg (by decide), innerCandidates293_14_checked]
  change scan edge (search edge 3) 4 [1949] 1949 innerTail293_15
    (if 3 ≤ innerCandidates293_14.length then search edge 3 [1949, 7901] innerCandidates293_14 9850 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan293_15

def innerTail293_13 : List Nat := 7727 :: innerTail293_14
def innerCandidates293_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates293_13_checked : candidates edge 4 1949 7727 innerTail293_14 best.bound = innerCandidates293_13 := by
  unfold candidates
  have row : edge 7727 = SearchData.row7727.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan293_13 : scan edge (search edge 3) 4 [1949] 1949 innerTail293_13 best = best := by
  rw [innerTail293_13, scan, if_neg (by decide), innerCandidates293_13_checked]
  change scan edge (search edge 3) 4 [1949] 1949 innerTail293_14
    (if 3 ≤ innerCandidates293_13.length then search edge 3 [1949, 7727] innerCandidates293_13 9676 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan293_14

def innerTail293_12 : List Nat := 6863 :: innerTail293_13
def innerCandidates293_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates293_12_checked : candidates edge 4 1949 6863 innerTail293_13 best.bound = innerCandidates293_12 := by
  unfold candidates
  have row : edge 6863 = SearchData.row6863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan293_12 : scan edge (search edge 3) 4 [1949] 1949 innerTail293_12 best = best := by
  rw [innerTail293_12, scan, if_neg (by decide), innerCandidates293_12_checked]
  change scan edge (search edge 3) 4 [1949] 1949 innerTail293_13
    (if 3 ≤ innerCandidates293_12.length then search edge 3 [1949, 6863] innerCandidates293_12 8812 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan293_13

def innerTail293_11 : List Nat := 5669 :: innerTail293_12
def innerCandidates293_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates293_11_checked : candidates edge 4 1949 5669 innerTail293_12 best.bound = innerCandidates293_11 := by
  unfold candidates
  have row : edge 5669 = SearchData.row5669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan293_11 : scan edge (search edge 3) 4 [1949] 1949 innerTail293_11 best = best := by
  rw [innerTail293_11, scan, if_neg (by decide), innerCandidates293_11_checked]
  change scan edge (search edge 3) 4 [1949] 1949 innerTail293_12
    (if 3 ≤ innerCandidates293_11.length then search edge 3 [1949, 5669] innerCandidates293_11 7618 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan293_12

def innerTail293_10 : List Nat := 4337 :: innerTail293_11
def innerCandidates293_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates293_10_checked : candidates edge 4 1949 4337 innerTail293_11 best.bound = innerCandidates293_10 := by
  unfold candidates
  have row : edge 4337 = SearchData.row4337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan293_10 : scan edge (search edge 3) 4 [1949] 1949 innerTail293_10 best = best := by
  rw [innerTail293_10, scan, if_neg (by decide), innerCandidates293_10_checked]
  change scan edge (search edge 3) 4 [1949] 1949 innerTail293_11
    (if 3 ≤ innerCandidates293_10.length then search edge 3 [1949, 4337] innerCandidates293_10 6286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan293_11

def innerTail293_9 : List Nat := 3989 :: innerTail293_10
def innerCandidates293_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates293_9_checked : candidates edge 4 1949 3989 innerTail293_10 best.bound = innerCandidates293_9 := by
  unfold candidates
  have row : edge 3989 = SearchData.row3989.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan293_9 : scan edge (search edge 3) 4 [1949] 1949 innerTail293_9 best = best := by
  rw [innerTail293_9, scan, if_neg (by decide), innerCandidates293_9_checked]
  change scan edge (search edge 3) 4 [1949] 1949 innerTail293_10
    (if 3 ≤ innerCandidates293_9.length then search edge 3 [1949, 3989] innerCandidates293_9 5938 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan293_10

def innerTail293_8 : List Nat := 3923 :: innerTail293_9
def innerCandidates293_8 : List Nat := [3989]
set_option maxRecDepth 4096 in
theorem innerCandidates293_8_checked : candidates edge 4 1949 3923 innerTail293_9 best.bound = innerCandidates293_8 := by
  unfold candidates
  have row : edge 3923 = SearchData.row3923.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan293_8 : scan edge (search edge 3) 4 [1949] 1949 innerTail293_8 best = best := by
  rw [innerTail293_8, scan, if_neg (by decide), innerCandidates293_8_checked]
  change scan edge (search edge 3) 4 [1949] 1949 innerTail293_9
    (if 3 ≤ innerCandidates293_8.length then search edge 3 [1949, 3923] innerCandidates293_8 5872 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan293_9

def innerTail293_7 : List Nat := 3911 :: innerTail293_8
def innerCandidates293_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates293_7_checked : candidates edge 4 1949 3911 innerTail293_8 best.bound = innerCandidates293_7 := by
  unfold candidates
  have row : edge 3911 = SearchData.row3911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan293_7 : scan edge (search edge 3) 4 [1949] 1949 innerTail293_7 best = best := by
  rw [innerTail293_7, scan, if_neg (by decide), innerCandidates293_7_checked]
  change scan edge (search edge 3) 4 [1949] 1949 innerTail293_8
    (if 3 ≤ innerCandidates293_7.length then search edge 3 [1949, 3911] innerCandidates293_7 5860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan293_8

def innerTail293_6 : List Nat := 2903 :: innerTail293_7
def innerCandidates293_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates293_6_checked : candidates edge 4 1949 2903 innerTail293_7 best.bound = innerCandidates293_6 := by
  unfold candidates
  have row : edge 2903 = SearchData.row2903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan293_6 : scan edge (search edge 3) 4 [1949] 1949 innerTail293_6 best = best := by
  rw [innerTail293_6, scan, if_neg (by decide), innerCandidates293_6_checked]
  change scan edge (search edge 3) 4 [1949] 1949 innerTail293_7
    (if 3 ≤ innerCandidates293_6.length then search edge 3 [1949, 2903] innerCandidates293_6 4852 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan293_7

def innerTail293_5 : List Nat := 2819 :: innerTail293_6
def innerCandidates293_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates293_5_checked : candidates edge 4 1949 2819 innerTail293_6 best.bound = innerCandidates293_5 := by
  unfold candidates
  have row : edge 2819 = SearchData.row2819.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan293_5 : scan edge (search edge 3) 4 [1949] 1949 innerTail293_5 best = best := by
  rw [innerTail293_5, scan, if_neg (by decide), innerCandidates293_5_checked]
  change scan edge (search edge 3) 4 [1949] 1949 innerTail293_6
    (if 3 ≤ innerCandidates293_5.length then search edge 3 [1949, 2819] innerCandidates293_5 4768 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan293_6

def innerTail293_4 : List Nat := 2789 :: innerTail293_5
def innerCandidates293_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates293_4_checked : candidates edge 4 1949 2789 innerTail293_5 best.bound = innerCandidates293_4 := by
  unfold candidates
  have row : edge 2789 = SearchData.row2789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan293_4 : scan edge (search edge 3) 4 [1949] 1949 innerTail293_4 best = best := by
  rw [innerTail293_4, scan, if_neg (by decide), innerCandidates293_4_checked]
  change scan edge (search edge 3) 4 [1949] 1949 innerTail293_5
    (if 3 ≤ innerCandidates293_4.length then search edge 3 [1949, 2789] innerCandidates293_4 4738 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan293_5

def innerTail293_3 : List Nat := 2663 :: innerTail293_4
def innerCandidates293_3 : List Nat := [5669, 7901]
set_option maxRecDepth 4096 in
theorem innerCandidates293_3_checked : candidates edge 4 1949 2663 innerTail293_4 best.bound = innerCandidates293_3 := by
  unfold candidates
  have row : edge 2663 = PrimePairCache.smallRow2663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan293_3 : scan edge (search edge 3) 4 [1949] 1949 innerTail293_3 best = best := by
  rw [innerTail293_3, scan, if_neg (by decide), innerCandidates293_3_checked]
  change scan edge (search edge 3) 4 [1949] 1949 innerTail293_4
    (if 3 ≤ innerCandidates293_3.length then search edge 3 [1949, 2663] innerCandidates293_3 4612 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan293_4

def innerTail293_2 : List Nat := 2657 :: innerTail293_3
def innerCandidates293_2 : List Nat := [3923]
set_option maxRecDepth 4096 in
theorem innerCandidates293_2_checked : candidates edge 4 1949 2657 innerTail293_3 best.bound = innerCandidates293_2 := by
  unfold candidates
  have row : edge 2657 = PrimePairCache.smallRow2657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan293_2 : scan edge (search edge 3) 4 [1949] 1949 innerTail293_2 best = best := by
  rw [innerTail293_2, scan, if_neg (by decide), innerCandidates293_2_checked]
  change scan edge (search edge 3) 4 [1949] 1949 innerTail293_3
    (if 3 ≤ innerCandidates293_2.length then search edge 3 [1949, 2657] innerCandidates293_2 4606 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan293_3

def innerTail293_1 : List Nat := 2621 :: innerTail293_2
def innerCandidates293_1 : List Nat := [2789]
set_option maxRecDepth 4096 in
theorem innerCandidates293_1_checked : candidates edge 4 1949 2621 innerTail293_2 best.bound = innerCandidates293_1 := by
  unfold candidates
  have row : edge 2621 = PrimePairCache.smallRow2621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan293_1 : scan edge (search edge 3) 4 [1949] 1949 innerTail293_1 best = best := by
  rw [innerTail293_1, scan, if_neg (by decide), innerCandidates293_1_checked]
  change scan edge (search edge 3) 4 [1949] 1949 innerTail293_2
    (if 3 ≤ innerCandidates293_1.length then search edge 3 [1949, 2621] innerCandidates293_1 4570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan293_2

def innerTail293_0 : List Nat := 2027 :: innerTail293_1
def innerCandidates293_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates293_0_checked : candidates edge 4 1949 2027 innerTail293_1 best.bound = innerCandidates293_0 := by
  unfold candidates
  have row : edge 2027 = PrimePairCache.smallRow2027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan293_0 : scan edge (search edge 3) 4 [1949] 1949 innerTail293_0 best = best := by
  rw [innerTail293_0, scan, if_neg (by decide), innerCandidates293_0_checked]
  change scan edge (search edge 3) 4 [1949] 1949 innerTail293_1
    (if 3 ≤ innerCandidates293_0.length then search edge 3 [1949, 2027] innerCandidates293_0 3976 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan293_1

theorem rootBranch293 : search edge 4 [1949] rootCandidates293 1949 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1949] 1949 innerTail293_0 best = best
  exact innerScan293_0

def rootCandidates292 : List Nat := [2293, 3313, 3733, 4201, 5011, 5023, 5791, 6703, 6841, 7333, 7639, 7933, 8287]
set_option maxRecDepth 4096 in
theorem rootCandidates292_checked : candidates edge 5 0 1933 tail293 best.bound = rootCandidates292 := by
  unfold candidates
  have row : edge 1933 = PrimePairCache.smallRow1933.contains := by funext q; rfl
  rw [row]
  decide

def innerTail292_13 : List Nat := []
theorem innerScan292_13 : scan edge (search edge 3) 4 [1933] 1933 innerTail292_13 best = best := rfl

def innerTail292_12 : List Nat := 8287 :: innerTail292_13
def innerCandidates292_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates292_12_checked : candidates edge 4 1933 8287 innerTail292_13 best.bound = innerCandidates292_12 := by
  unfold candidates
  have row : edge 8287 = SearchData.row8287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan292_12 : scan edge (search edge 3) 4 [1933] 1933 innerTail292_12 best = best := by
  rw [innerTail292_12, scan, if_neg (by decide), innerCandidates292_12_checked]
  change scan edge (search edge 3) 4 [1933] 1933 innerTail292_13
    (if 3 ≤ innerCandidates292_12.length then search edge 3 [1933, 8287] innerCandidates292_12 10220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan292_13

def innerTail292_11 : List Nat := 7933 :: innerTail292_12
def innerCandidates292_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates292_11_checked : candidates edge 4 1933 7933 innerTail292_12 best.bound = innerCandidates292_11 := by
  unfold candidates
  have row : edge 7933 = SearchData.row7933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan292_11 : scan edge (search edge 3) 4 [1933] 1933 innerTail292_11 best = best := by
  rw [innerTail292_11, scan, if_neg (by decide), innerCandidates292_11_checked]
  change scan edge (search edge 3) 4 [1933] 1933 innerTail292_12
    (if 3 ≤ innerCandidates292_11.length then search edge 3 [1933, 7933] innerCandidates292_11 9866 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan292_12

def innerTail292_10 : List Nat := 7639 :: innerTail292_11
def innerCandidates292_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates292_10_checked : candidates edge 4 1933 7639 innerTail292_11 best.bound = innerCandidates292_10 := by
  unfold candidates
  have row : edge 7639 = SearchData.row7639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan292_10 : scan edge (search edge 3) 4 [1933] 1933 innerTail292_10 best = best := by
  rw [innerTail292_10, scan, if_neg (by decide), innerCandidates292_10_checked]
  change scan edge (search edge 3) 4 [1933] 1933 innerTail292_11
    (if 3 ≤ innerCandidates292_10.length then search edge 3 [1933, 7639] innerCandidates292_10 9572 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan292_11

def innerTail292_9 : List Nat := 7333 :: innerTail292_10
def innerCandidates292_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates292_9_checked : candidates edge 4 1933 7333 innerTail292_10 best.bound = innerCandidates292_9 := by
  unfold candidates
  have row : edge 7333 = SearchData.row7333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan292_9 : scan edge (search edge 3) 4 [1933] 1933 innerTail292_9 best = best := by
  rw [innerTail292_9, scan, if_neg (by decide), innerCandidates292_9_checked]
  change scan edge (search edge 3) 4 [1933] 1933 innerTail292_10
    (if 3 ≤ innerCandidates292_9.length then search edge 3 [1933, 7333] innerCandidates292_9 9266 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan292_10

def innerTail292_8 : List Nat := 6841 :: innerTail292_9
def innerCandidates292_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates292_8_checked : candidates edge 4 1933 6841 innerTail292_9 best.bound = innerCandidates292_8 := by
  unfold candidates
  have row : edge 6841 = SearchData.row6841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan292_8 : scan edge (search edge 3) 4 [1933] 1933 innerTail292_8 best = best := by
  rw [innerTail292_8, scan, if_neg (by decide), innerCandidates292_8_checked]
  change scan edge (search edge 3) 4 [1933] 1933 innerTail292_9
    (if 3 ≤ innerCandidates292_8.length then search edge 3 [1933, 6841] innerCandidates292_8 8774 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan292_9

def innerTail292_7 : List Nat := 6703 :: innerTail292_8
def innerCandidates292_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates292_7_checked : candidates edge 4 1933 6703 innerTail292_8 best.bound = innerCandidates292_7 := by
  unfold candidates
  have row : edge 6703 = SearchData.row6703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan292_7 : scan edge (search edge 3) 4 [1933] 1933 innerTail292_7 best = best := by
  rw [innerTail292_7, scan, if_neg (by decide), innerCandidates292_7_checked]
  change scan edge (search edge 3) 4 [1933] 1933 innerTail292_8
    (if 3 ≤ innerCandidates292_7.length then search edge 3 [1933, 6703] innerCandidates292_7 8636 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan292_8

def innerTail292_6 : List Nat := 5791 :: innerTail292_7
def innerCandidates292_6 : List Nat := [7333]
set_option maxRecDepth 4096 in
theorem innerCandidates292_6_checked : candidates edge 4 1933 5791 innerTail292_7 best.bound = innerCandidates292_6 := by
  unfold candidates
  have row : edge 5791 = SearchData.row5791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan292_6 : scan edge (search edge 3) 4 [1933] 1933 innerTail292_6 best = best := by
  rw [innerTail292_6, scan, if_neg (by decide), innerCandidates292_6_checked]
  change scan edge (search edge 3) 4 [1933] 1933 innerTail292_7
    (if 3 ≤ innerCandidates292_6.length then search edge 3 [1933, 5791] innerCandidates292_6 7724 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan292_7

def innerTail292_5 : List Nat := 5023 :: innerTail292_6
def innerCandidates292_5 : List Nat := [6841]
set_option maxRecDepth 4096 in
theorem innerCandidates292_5_checked : candidates edge 4 1933 5023 innerTail292_6 best.bound = innerCandidates292_5 := by
  unfold candidates
  have row : edge 5023 = SearchData.row5023.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan292_5 : scan edge (search edge 3) 4 [1933] 1933 innerTail292_5 best = best := by
  rw [innerTail292_5, scan, if_neg (by decide), innerCandidates292_5_checked]
  change scan edge (search edge 3) 4 [1933] 1933 innerTail292_6
    (if 3 ≤ innerCandidates292_5.length then search edge 3 [1933, 5023] innerCandidates292_5 6956 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan292_6

def innerTail292_4 : List Nat := 5011 :: innerTail292_5
def innerCandidates292_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates292_4_checked : candidates edge 4 1933 5011 innerTail292_5 best.bound = innerCandidates292_4 := by
  unfold candidates
  have row : edge 5011 = SearchData.row5011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan292_4 : scan edge (search edge 3) 4 [1933] 1933 innerTail292_4 best = best := by
  rw [innerTail292_4, scan, if_neg (by decide), innerCandidates292_4_checked]
  change scan edge (search edge 3) 4 [1933] 1933 innerTail292_5
    (if 3 ≤ innerCandidates292_4.length then search edge 3 [1933, 5011] innerCandidates292_4 6944 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan292_5

def innerTail292_3 : List Nat := 4201 :: innerTail292_4
def innerCandidates292_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates292_3_checked : candidates edge 4 1933 4201 innerTail292_4 best.bound = innerCandidates292_3 := by
  unfold candidates
  have row : edge 4201 = SearchData.row4201.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan292_3 : scan edge (search edge 3) 4 [1933] 1933 innerTail292_3 best = best := by
  rw [innerTail292_3, scan, if_neg (by decide), innerCandidates292_3_checked]
  change scan edge (search edge 3) 4 [1933] 1933 innerTail292_4
    (if 3 ≤ innerCandidates292_3.length then search edge 3 [1933, 4201] innerCandidates292_3 6134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan292_4

def innerTail292_2 : List Nat := 3733 :: innerTail292_3
def innerCandidates292_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates292_2_checked : candidates edge 4 1933 3733 innerTail292_3 best.bound = innerCandidates292_2 := by
  unfold candidates
  have row : edge 3733 = SearchData.row3733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan292_2 : scan edge (search edge 3) 4 [1933] 1933 innerTail292_2 best = best := by
  rw [innerTail292_2, scan, if_neg (by decide), innerCandidates292_2_checked]
  change scan edge (search edge 3) 4 [1933] 1933 innerTail292_3
    (if 3 ≤ innerCandidates292_2.length then search edge 3 [1933, 3733] innerCandidates292_2 5666 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan292_3

def innerTail292_1 : List Nat := 3313 :: innerTail292_2
def innerCandidates292_1 : List Nat := [3733]
set_option maxRecDepth 4096 in
theorem innerCandidates292_1_checked : candidates edge 4 1933 3313 innerTail292_2 best.bound = innerCandidates292_1 := by
  unfold candidates
  have row : edge 3313 = SearchData.row3313.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan292_1 : scan edge (search edge 3) 4 [1933] 1933 innerTail292_1 best = best := by
  rw [innerTail292_1, scan, if_neg (by decide), innerCandidates292_1_checked]
  change scan edge (search edge 3) 4 [1933] 1933 innerTail292_2
    (if 3 ≤ innerCandidates292_1.length then search edge 3 [1933, 3313] innerCandidates292_1 5246 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan292_2

def innerTail292_0 : List Nat := 2293 :: innerTail292_1
def innerCandidates292_0 : List Nat := [7333]
set_option maxRecDepth 4096 in
theorem innerCandidates292_0_checked : candidates edge 4 1933 2293 innerTail292_1 best.bound = innerCandidates292_0 := by
  unfold candidates
  have row : edge 2293 = PrimePairCache.smallRow2293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan292_0 : scan edge (search edge 3) 4 [1933] 1933 innerTail292_0 best = best := by
  rw [innerTail292_0, scan, if_neg (by decide), innerCandidates292_0_checked]
  change scan edge (search edge 3) 4 [1933] 1933 innerTail292_1
    (if 3 ≤ innerCandidates292_0.length then search edge 3 [1933, 2293] innerCandidates292_0 4226 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan292_1

theorem rootBranch292 : search edge 4 [1933] rootCandidates292 1933 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1933] 1933 innerTail292_0 best = best
  exact innerScan292_0

end ProjectEulerSolutions.PrimePairUpper.Search
