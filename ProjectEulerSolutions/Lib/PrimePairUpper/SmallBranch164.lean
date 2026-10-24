import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates186 : List Nat := [1741, 1999, 2143, 2269, 2473, 2557, 2647, 2767, 3001, 4561, 5107, 5923, 6421, 6451, 6829, 6871, 7057, 7417, 7603, 7621, 8233]
set_option maxRecDepth 4096 in
theorem rootCandidates186_checked : candidates edge 5 0 1129 tail187 best.bound = rootCandidates186 := by
  unfold candidates
  have row : edge 1129 = PrimePairCache.smallRow1129.contains := by funext q; rfl
  rw [row]
  decide

def innerTail186_21 : List Nat := []
theorem innerScan186_21 : scan edge (search edge 3) 4 [1129] 1129 innerTail186_21 best = best := rfl

def innerTail186_20 : List Nat := 8233 :: innerTail186_21
def innerCandidates186_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates186_20_checked : candidates edge 4 1129 8233 innerTail186_21 best.bound = innerCandidates186_20 := by
  unfold candidates
  have row : edge 8233 = SearchData.row8233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan186_20 : scan edge (search edge 3) 4 [1129] 1129 innerTail186_20 best = best := by
  rw [innerTail186_20, scan, if_neg (by decide), innerCandidates186_20_checked]
  change scan edge (search edge 3) 4 [1129] 1129 innerTail186_21
    (if 3 ≤ innerCandidates186_20.length then search edge 3 [1129, 8233] innerCandidates186_20 9362 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan186_21

def innerTail186_19 : List Nat := 7621 :: innerTail186_20
def innerCandidates186_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates186_19_checked : candidates edge 4 1129 7621 innerTail186_20 best.bound = innerCandidates186_19 := by
  unfold candidates
  have row : edge 7621 = SearchData.row7621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan186_19 : scan edge (search edge 3) 4 [1129] 1129 innerTail186_19 best = best := by
  rw [innerTail186_19, scan, if_neg (by decide), innerCandidates186_19_checked]
  change scan edge (search edge 3) 4 [1129] 1129 innerTail186_20
    (if 3 ≤ innerCandidates186_19.length then search edge 3 [1129, 7621] innerCandidates186_19 8750 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan186_20

def innerTail186_18 : List Nat := 7603 :: innerTail186_19
def innerCandidates186_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates186_18_checked : candidates edge 4 1129 7603 innerTail186_19 best.bound = innerCandidates186_18 := by
  unfold candidates
  have row : edge 7603 = SearchData.row7603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan186_18 : scan edge (search edge 3) 4 [1129] 1129 innerTail186_18 best = best := by
  rw [innerTail186_18, scan, if_neg (by decide), innerCandidates186_18_checked]
  change scan edge (search edge 3) 4 [1129] 1129 innerTail186_19
    (if 3 ≤ innerCandidates186_18.length then search edge 3 [1129, 7603] innerCandidates186_18 8732 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan186_19

def innerTail186_17 : List Nat := 7417 :: innerTail186_18
def innerCandidates186_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates186_17_checked : candidates edge 4 1129 7417 innerTail186_18 best.bound = innerCandidates186_17 := by
  unfold candidates
  have row : edge 7417 = SearchData.row7417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan186_17 : scan edge (search edge 3) 4 [1129] 1129 innerTail186_17 best = best := by
  rw [innerTail186_17, scan, if_neg (by decide), innerCandidates186_17_checked]
  change scan edge (search edge 3) 4 [1129] 1129 innerTail186_18
    (if 3 ≤ innerCandidates186_17.length then search edge 3 [1129, 7417] innerCandidates186_17 8546 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan186_18

def innerTail186_16 : List Nat := 7057 :: innerTail186_17
def innerCandidates186_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates186_16_checked : candidates edge 4 1129 7057 innerTail186_17 best.bound = innerCandidates186_16 := by
  unfold candidates
  have row : edge 7057 = SearchData.row7057.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan186_16 : scan edge (search edge 3) 4 [1129] 1129 innerTail186_16 best = best := by
  rw [innerTail186_16, scan, if_neg (by decide), innerCandidates186_16_checked]
  change scan edge (search edge 3) 4 [1129] 1129 innerTail186_17
    (if 3 ≤ innerCandidates186_16.length then search edge 3 [1129, 7057] innerCandidates186_16 8186 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan186_17

def innerTail186_15 : List Nat := 6871 :: innerTail186_16
def innerCandidates186_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates186_15_checked : candidates edge 4 1129 6871 innerTail186_16 best.bound = innerCandidates186_15 := by
  unfold candidates
  have row : edge 6871 = SearchData.row6871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan186_15 : scan edge (search edge 3) 4 [1129] 1129 innerTail186_15 best = best := by
  rw [innerTail186_15, scan, if_neg (by decide), innerCandidates186_15_checked]
  change scan edge (search edge 3) 4 [1129] 1129 innerTail186_16
    (if 3 ≤ innerCandidates186_15.length then search edge 3 [1129, 6871] innerCandidates186_15 8000 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan186_16

def innerTail186_14 : List Nat := 6829 :: innerTail186_15
def innerCandidates186_14 : List Nat := [7057]
set_option maxRecDepth 4096 in
theorem innerCandidates186_14_checked : candidates edge 4 1129 6829 innerTail186_15 best.bound = innerCandidates186_14 := by
  unfold candidates
  have row : edge 6829 = SearchData.row6829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan186_14 : scan edge (search edge 3) 4 [1129] 1129 innerTail186_14 best = best := by
  rw [innerTail186_14, scan, if_neg (by decide), innerCandidates186_14_checked]
  change scan edge (search edge 3) 4 [1129] 1129 innerTail186_15
    (if 3 ≤ innerCandidates186_14.length then search edge 3 [1129, 6829] innerCandidates186_14 7958 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan186_15

def innerTail186_13 : List Nat := 6451 :: innerTail186_14
def innerCandidates186_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates186_13_checked : candidates edge 4 1129 6451 innerTail186_14 best.bound = innerCandidates186_13 := by
  unfold candidates
  have row : edge 6451 = SearchData.row6451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan186_13 : scan edge (search edge 3) 4 [1129] 1129 innerTail186_13 best = best := by
  rw [innerTail186_13, scan, if_neg (by decide), innerCandidates186_13_checked]
  change scan edge (search edge 3) 4 [1129] 1129 innerTail186_14
    (if 3 ≤ innerCandidates186_13.length then search edge 3 [1129, 6451] innerCandidates186_13 7580 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan186_14

def innerTail186_12 : List Nat := 6421 :: innerTail186_13
def innerCandidates186_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates186_12_checked : candidates edge 4 1129 6421 innerTail186_13 best.bound = innerCandidates186_12 := by
  unfold candidates
  have row : edge 6421 = SearchData.row6421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan186_12 : scan edge (search edge 3) 4 [1129] 1129 innerTail186_12 best = best := by
  rw [innerTail186_12, scan, if_neg (by decide), innerCandidates186_12_checked]
  change scan edge (search edge 3) 4 [1129] 1129 innerTail186_13
    (if 3 ≤ innerCandidates186_12.length then search edge 3 [1129, 6421] innerCandidates186_12 7550 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan186_13

def innerTail186_11 : List Nat := 5923 :: innerTail186_12
def innerCandidates186_11 : List Nat := [6421, 6871, 7417]
set_option maxRecDepth 4096 in
theorem innerCandidates186_11_checked : candidates edge 4 1129 5923 innerTail186_12 best.bound = innerCandidates186_11 := by
  unfold candidates
  have row : edge 5923 = SearchData.row5923.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild186_11 : search edge 3 [1129, 5923] innerCandidates186_11 7052 best = best := by decide

theorem innerScan186_11 : scan edge (search edge 3) 4 [1129] 1129 innerTail186_11 best = best := by
  rw [innerTail186_11, scan, if_neg (by decide), innerCandidates186_11_checked]
  change scan edge (search edge 3) 4 [1129] 1129 innerTail186_12
    (if 3 ≤ innerCandidates186_11.length then search edge 3 [1129, 5923] innerCandidates186_11 7052 best else best) = best
  rw [if_pos (by decide), innerChild186_11]
  exact innerScan186_12

def innerTail186_10 : List Nat := 5107 :: innerTail186_11
def innerCandidates186_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates186_10_checked : candidates edge 4 1129 5107 innerTail186_11 best.bound = innerCandidates186_10 := by
  unfold candidates
  have row : edge 5107 = SearchData.row5107.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan186_10 : scan edge (search edge 3) 4 [1129] 1129 innerTail186_10 best = best := by
  rw [innerTail186_10, scan, if_neg (by decide), innerCandidates186_10_checked]
  change scan edge (search edge 3) 4 [1129] 1129 innerTail186_11
    (if 3 ≤ innerCandidates186_10.length then search edge 3 [1129, 5107] innerCandidates186_10 6236 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan186_11

def innerTail186_9 : List Nat := 4561 :: innerTail186_10
def innerCandidates186_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates186_9_checked : candidates edge 4 1129 4561 innerTail186_10 best.bound = innerCandidates186_9 := by
  unfold candidates
  have row : edge 4561 = SearchData.row4561.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan186_9 : scan edge (search edge 3) 4 [1129] 1129 innerTail186_9 best = best := by
  rw [innerTail186_9, scan, if_neg (by decide), innerCandidates186_9_checked]
  change scan edge (search edge 3) 4 [1129] 1129 innerTail186_10
    (if 3 ≤ innerCandidates186_9.length then search edge 3 [1129, 4561] innerCandidates186_9 5690 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan186_10

def innerTail186_8 : List Nat := 3001 :: innerTail186_9
def innerCandidates186_8 : List Nat := [7621]
set_option maxRecDepth 4096 in
theorem innerCandidates186_8_checked : candidates edge 4 1129 3001 innerTail186_9 best.bound = innerCandidates186_8 := by
  unfold candidates
  have row : edge 3001 = SearchData.row3001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan186_8 : scan edge (search edge 3) 4 [1129] 1129 innerTail186_8 best = best := by
  rw [innerTail186_8, scan, if_neg (by decide), innerCandidates186_8_checked]
  change scan edge (search edge 3) 4 [1129] 1129 innerTail186_9
    (if 3 ≤ innerCandidates186_8.length then search edge 3 [1129, 3001] innerCandidates186_8 4130 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan186_9

def innerTail186_7 : List Nat := 2767 :: innerTail186_8
def innerCandidates186_7 : List Nat := [5107, 7621]
set_option maxRecDepth 4096 in
theorem innerCandidates186_7_checked : candidates edge 4 1129 2767 innerTail186_8 best.bound = innerCandidates186_7 := by
  unfold candidates
  have row : edge 2767 = SearchData.row2767.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan186_7 : scan edge (search edge 3) 4 [1129] 1129 innerTail186_7 best = best := by
  rw [innerTail186_7, scan, if_neg (by decide), innerCandidates186_7_checked]
  change scan edge (search edge 3) 4 [1129] 1129 innerTail186_8
    (if 3 ≤ innerCandidates186_7.length then search edge 3 [1129, 2767] innerCandidates186_7 3896 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan186_8

def innerTail186_6 : List Nat := 2647 :: innerTail186_7
def innerCandidates186_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates186_6_checked : candidates edge 4 1129 2647 innerTail186_7 best.bound = innerCandidates186_6 := by
  unfold candidates
  have row : edge 2647 = PrimePairCache.smallRow2647.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan186_6 : scan edge (search edge 3) 4 [1129] 1129 innerTail186_6 best = best := by
  rw [innerTail186_6, scan, if_neg (by decide), innerCandidates186_6_checked]
  change scan edge (search edge 3) 4 [1129] 1129 innerTail186_7
    (if 3 ≤ innerCandidates186_6.length then search edge 3 [1129, 2647] innerCandidates186_6 3776 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan186_7

def innerTail186_5 : List Nat := 2557 :: innerTail186_6
def innerCandidates186_5 : List Nat := [6871]
set_option maxRecDepth 4096 in
theorem innerCandidates186_5_checked : candidates edge 4 1129 2557 innerTail186_6 best.bound = innerCandidates186_5 := by
  unfold candidates
  have row : edge 2557 = PrimePairCache.smallRow2557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan186_5 : scan edge (search edge 3) 4 [1129] 1129 innerTail186_5 best = best := by
  rw [innerTail186_5, scan, if_neg (by decide), innerCandidates186_5_checked]
  change scan edge (search edge 3) 4 [1129] 1129 innerTail186_6
    (if 3 ≤ innerCandidates186_5.length then search edge 3 [1129, 2557] innerCandidates186_5 3686 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan186_6

def innerTail186_4 : List Nat := 2473 :: innerTail186_5
def innerCandidates186_4 : List Nat := [2647, 3001, 7417]
set_option maxRecDepth 4096 in
theorem innerCandidates186_4_checked : candidates edge 4 1129 2473 innerTail186_5 best.bound = innerCandidates186_4 := by
  unfold candidates
  have row : edge 2473 = PrimePairCache.smallRow2473.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild186_4 : search edge 3 [1129, 2473] innerCandidates186_4 3602 best = best := by decide

theorem innerScan186_4 : scan edge (search edge 3) 4 [1129] 1129 innerTail186_4 best = best := by
  rw [innerTail186_4, scan, if_neg (by decide), innerCandidates186_4_checked]
  change scan edge (search edge 3) 4 [1129] 1129 innerTail186_5
    (if 3 ≤ innerCandidates186_4.length then search edge 3 [1129, 2473] innerCandidates186_4 3602 best else best) = best
  rw [if_pos (by decide), innerChild186_4]
  exact innerScan186_5

def innerTail186_3 : List Nat := 2269 :: innerTail186_4
def innerCandidates186_3 : List Nat := [5107, 5923]
set_option maxRecDepth 4096 in
theorem innerCandidates186_3_checked : candidates edge 4 1129 2269 innerTail186_4 best.bound = innerCandidates186_3 := by
  unfold candidates
  have row : edge 2269 = PrimePairCache.smallRow2269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan186_3 : scan edge (search edge 3) 4 [1129] 1129 innerTail186_3 best = best := by
  rw [innerTail186_3, scan, if_neg (by decide), innerCandidates186_3_checked]
  change scan edge (search edge 3) 4 [1129] 1129 innerTail186_4
    (if 3 ≤ innerCandidates186_3.length then search edge 3 [1129, 2269] innerCandidates186_3 3398 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan186_4

def innerTail186_2 : List Nat := 2143 :: innerTail186_3
def innerCandidates186_2 : List Nat := [6871]
set_option maxRecDepth 4096 in
theorem innerCandidates186_2_checked : candidates edge 4 1129 2143 innerTail186_3 best.bound = innerCandidates186_2 := by
  unfold candidates
  have row : edge 2143 = PrimePairCache.smallRow2143.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan186_2 : scan edge (search edge 3) 4 [1129] 1129 innerTail186_2 best = best := by
  rw [innerTail186_2, scan, if_neg (by decide), innerCandidates186_2_checked]
  change scan edge (search edge 3) 4 [1129] 1129 innerTail186_3
    (if 3 ≤ innerCandidates186_2.length then search edge 3 [1129, 2143] innerCandidates186_2 3272 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan186_3

def innerTail186_1 : List Nat := 1999 :: innerTail186_2
def innerCandidates186_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates186_1_checked : candidates edge 4 1129 1999 innerTail186_2 best.bound = innerCandidates186_1 := by
  unfold candidates
  have row : edge 1999 = PrimePairCache.smallRow1999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan186_1 : scan edge (search edge 3) 4 [1129] 1129 innerTail186_1 best = best := by
  rw [innerTail186_1, scan, if_neg (by decide), innerCandidates186_1_checked]
  change scan edge (search edge 3) 4 [1129] 1129 innerTail186_2
    (if 3 ≤ innerCandidates186_1.length then search edge 3 [1129, 1999] innerCandidates186_1 3128 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan186_2

def innerTail186_0 : List Nat := 1741 :: innerTail186_1
def innerCandidates186_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates186_0_checked : candidates edge 4 1129 1741 innerTail186_1 best.bound = innerCandidates186_0 := by
  unfold candidates
  have row : edge 1741 = PrimePairCache.smallRow1741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan186_0 : scan edge (search edge 3) 4 [1129] 1129 innerTail186_0 best = best := by
  rw [innerTail186_0, scan, if_neg (by decide), innerCandidates186_0_checked]
  change scan edge (search edge 3) 4 [1129] 1129 innerTail186_1
    (if 3 ≤ innerCandidates186_0.length then search edge 3 [1129, 1741] innerCandidates186_0 2870 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan186_1

theorem rootBranch186 : search edge 4 [1129] rootCandidates186 1129 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1129] 1129 innerTail186_0 best = best
  exact innerScan186_0

def rootCandidates185 : List Nat := [1201, 1399, 1453, 1657, 1783, 2029, 2203, 2281, 3373, 3433, 3919, 4423, 5281, 5347, 5527, 5641, 5737, 6037, 7489, 8221]
set_option maxRecDepth 4096 in
theorem rootCandidates185_checked : candidates edge 5 0 1123 tail186 best.bound = rootCandidates185 := by
  unfold candidates
  have row : edge 1123 = PrimePairCache.smallRow1123.contains := by funext q; rfl
  rw [row]
  decide

def innerTail185_20 : List Nat := []
theorem innerScan185_20 : scan edge (search edge 3) 4 [1123] 1123 innerTail185_20 best = best := rfl

def innerTail185_19 : List Nat := 8221 :: innerTail185_20
def innerCandidates185_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates185_19_checked : candidates edge 4 1123 8221 innerTail185_20 best.bound = innerCandidates185_19 := by
  unfold candidates
  have row : edge 8221 = SearchData.row8221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan185_19 : scan edge (search edge 3) 4 [1123] 1123 innerTail185_19 best = best := by
  rw [innerTail185_19, scan, if_neg (by decide), innerCandidates185_19_checked]
  change scan edge (search edge 3) 4 [1123] 1123 innerTail185_20
    (if 3 ≤ innerCandidates185_19.length then search edge 3 [1123, 8221] innerCandidates185_19 9344 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan185_20

def innerTail185_18 : List Nat := 7489 :: innerTail185_19
def innerCandidates185_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates185_18_checked : candidates edge 4 1123 7489 innerTail185_19 best.bound = innerCandidates185_18 := by
  unfold candidates
  have row : edge 7489 = SearchData.row7489.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan185_18 : scan edge (search edge 3) 4 [1123] 1123 innerTail185_18 best = best := by
  rw [innerTail185_18, scan, if_neg (by decide), innerCandidates185_18_checked]
  change scan edge (search edge 3) 4 [1123] 1123 innerTail185_19
    (if 3 ≤ innerCandidates185_18.length then search edge 3 [1123, 7489] innerCandidates185_18 8612 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan185_19

def innerTail185_17 : List Nat := 6037 :: innerTail185_18
def innerCandidates185_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates185_17_checked : candidates edge 4 1123 6037 innerTail185_18 best.bound = innerCandidates185_17 := by
  unfold candidates
  have row : edge 6037 = SearchData.row6037.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan185_17 : scan edge (search edge 3) 4 [1123] 1123 innerTail185_17 best = best := by
  rw [innerTail185_17, scan, if_neg (by decide), innerCandidates185_17_checked]
  change scan edge (search edge 3) 4 [1123] 1123 innerTail185_18
    (if 3 ≤ innerCandidates185_17.length then search edge 3 [1123, 6037] innerCandidates185_17 7160 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan185_18

def innerTail185_16 : List Nat := 5737 :: innerTail185_17
def innerCandidates185_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates185_16_checked : candidates edge 4 1123 5737 innerTail185_17 best.bound = innerCandidates185_16 := by
  unfold candidates
  have row : edge 5737 = SearchData.row5737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan185_16 : scan edge (search edge 3) 4 [1123] 1123 innerTail185_16 best = best := by
  rw [innerTail185_16, scan, if_neg (by decide), innerCandidates185_16_checked]
  change scan edge (search edge 3) 4 [1123] 1123 innerTail185_17
    (if 3 ≤ innerCandidates185_16.length then search edge 3 [1123, 5737] innerCandidates185_16 6860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan185_17

def innerTail185_15 : List Nat := 5641 :: innerTail185_16
def innerCandidates185_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates185_15_checked : candidates edge 4 1123 5641 innerTail185_16 best.bound = innerCandidates185_15 := by
  unfold candidates
  have row : edge 5641 = SearchData.row5641.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan185_15 : scan edge (search edge 3) 4 [1123] 1123 innerTail185_15 best = best := by
  rw [innerTail185_15, scan, if_neg (by decide), innerCandidates185_15_checked]
  change scan edge (search edge 3) 4 [1123] 1123 innerTail185_16
    (if 3 ≤ innerCandidates185_15.length then search edge 3 [1123, 5641] innerCandidates185_15 6764 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan185_16

def innerTail185_14 : List Nat := 5527 :: innerTail185_15
def innerCandidates185_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates185_14_checked : candidates edge 4 1123 5527 innerTail185_15 best.bound = innerCandidates185_14 := by
  unfold candidates
  have row : edge 5527 = SearchData.row5527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan185_14 : scan edge (search edge 3) 4 [1123] 1123 innerTail185_14 best = best := by
  rw [innerTail185_14, scan, if_neg (by decide), innerCandidates185_14_checked]
  change scan edge (search edge 3) 4 [1123] 1123 innerTail185_15
    (if 3 ≤ innerCandidates185_14.length then search edge 3 [1123, 5527] innerCandidates185_14 6650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan185_15

def innerTail185_13 : List Nat := 5347 :: innerTail185_14
def innerCandidates185_13 : List Nat := [8221]
set_option maxRecDepth 4096 in
theorem innerCandidates185_13_checked : candidates edge 4 1123 5347 innerTail185_14 best.bound = innerCandidates185_13 := by
  unfold candidates
  have row : edge 5347 = SearchData.row5347.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan185_13 : scan edge (search edge 3) 4 [1123] 1123 innerTail185_13 best = best := by
  rw [innerTail185_13, scan, if_neg (by decide), innerCandidates185_13_checked]
  change scan edge (search edge 3) 4 [1123] 1123 innerTail185_14
    (if 3 ≤ innerCandidates185_13.length then search edge 3 [1123, 5347] innerCandidates185_13 6470 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan185_14

def innerTail185_12 : List Nat := 5281 :: innerTail185_13
def innerCandidates185_12 : List Nat := [7489]
set_option maxRecDepth 4096 in
theorem innerCandidates185_12_checked : candidates edge 4 1123 5281 innerTail185_13 best.bound = innerCandidates185_12 := by
  unfold candidates
  have row : edge 5281 = SearchData.row5281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan185_12 : scan edge (search edge 3) 4 [1123] 1123 innerTail185_12 best = best := by
  rw [innerTail185_12, scan, if_neg (by decide), innerCandidates185_12_checked]
  change scan edge (search edge 3) 4 [1123] 1123 innerTail185_13
    (if 3 ≤ innerCandidates185_12.length then search edge 3 [1123, 5281] innerCandidates185_12 6404 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan185_13

def innerTail185_11 : List Nat := 4423 :: innerTail185_12
def innerCandidates185_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates185_11_checked : candidates edge 4 1123 4423 innerTail185_12 best.bound = innerCandidates185_11 := by
  unfold candidates
  have row : edge 4423 = SearchData.row4423.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan185_11 : scan edge (search edge 3) 4 [1123] 1123 innerTail185_11 best = best := by
  rw [innerTail185_11, scan, if_neg (by decide), innerCandidates185_11_checked]
  change scan edge (search edge 3) 4 [1123] 1123 innerTail185_12
    (if 3 ≤ innerCandidates185_11.length then search edge 3 [1123, 4423] innerCandidates185_11 5546 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan185_12

def innerTail185_10 : List Nat := 3919 :: innerTail185_11
def innerCandidates185_10 : List Nat := [5281, 7489]
set_option maxRecDepth 4096 in
theorem innerCandidates185_10_checked : candidates edge 4 1123 3919 innerTail185_11 best.bound = innerCandidates185_10 := by
  unfold candidates
  have row : edge 3919 = SearchData.row3919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan185_10 : scan edge (search edge 3) 4 [1123] 1123 innerTail185_10 best = best := by
  rw [innerTail185_10, scan, if_neg (by decide), innerCandidates185_10_checked]
  change scan edge (search edge 3) 4 [1123] 1123 innerTail185_11
    (if 3 ≤ innerCandidates185_10.length then search edge 3 [1123, 3919] innerCandidates185_10 5042 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan185_11

def innerTail185_9 : List Nat := 3433 :: innerTail185_10
def innerCandidates185_9 : List Nat := [8221]
set_option maxRecDepth 4096 in
theorem innerCandidates185_9_checked : candidates edge 4 1123 3433 innerTail185_10 best.bound = innerCandidates185_9 := by
  unfold candidates
  have row : edge 3433 = SearchData.row3433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan185_9 : scan edge (search edge 3) 4 [1123] 1123 innerTail185_9 best = best := by
  rw [innerTail185_9, scan, if_neg (by decide), innerCandidates185_9_checked]
  change scan edge (search edge 3) 4 [1123] 1123 innerTail185_10
    (if 3 ≤ innerCandidates185_9.length then search edge 3 [1123, 3433] innerCandidates185_9 4556 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan185_10

def innerTail185_8 : List Nat := 3373 :: innerTail185_9
def innerCandidates185_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates185_8_checked : candidates edge 4 1123 3373 innerTail185_9 best.bound = innerCandidates185_8 := by
  unfold candidates
  have row : edge 3373 = SearchData.row3373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan185_8 : scan edge (search edge 3) 4 [1123] 1123 innerTail185_8 best = best := by
  rw [innerTail185_8, scan, if_neg (by decide), innerCandidates185_8_checked]
  change scan edge (search edge 3) 4 [1123] 1123 innerTail185_9
    (if 3 ≤ innerCandidates185_8.length then search edge 3 [1123, 3373] innerCandidates185_8 4496 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan185_9

def innerTail185_7 : List Nat := 2281 :: innerTail185_8
def innerCandidates185_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates185_7_checked : candidates edge 4 1123 2281 innerTail185_8 best.bound = innerCandidates185_7 := by
  unfold candidates
  have row : edge 2281 = PrimePairCache.smallRow2281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan185_7 : scan edge (search edge 3) 4 [1123] 1123 innerTail185_7 best = best := by
  rw [innerTail185_7, scan, if_neg (by decide), innerCandidates185_7_checked]
  change scan edge (search edge 3) 4 [1123] 1123 innerTail185_8
    (if 3 ≤ innerCandidates185_7.length then search edge 3 [1123, 2281] innerCandidates185_7 3404 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan185_8

def innerTail185_6 : List Nat := 2203 :: innerTail185_7
def innerCandidates185_6 : List Nat := [6037]
set_option maxRecDepth 4096 in
theorem innerCandidates185_6_checked : candidates edge 4 1123 2203 innerTail185_7 best.bound = innerCandidates185_6 := by
  unfold candidates
  have row : edge 2203 = PrimePairCache.smallRow2203.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan185_6 : scan edge (search edge 3) 4 [1123] 1123 innerTail185_6 best = best := by
  rw [innerTail185_6, scan, if_neg (by decide), innerCandidates185_6_checked]
  change scan edge (search edge 3) 4 [1123] 1123 innerTail185_7
    (if 3 ≤ innerCandidates185_6.length then search edge 3 [1123, 2203] innerCandidates185_6 3326 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan185_7

def innerTail185_5 : List Nat := 2029 :: innerTail185_6
def innerCandidates185_5 : List Nat := [2281, 5281]
set_option maxRecDepth 4096 in
theorem innerCandidates185_5_checked : candidates edge 4 1123 2029 innerTail185_6 best.bound = innerCandidates185_5 := by
  unfold candidates
  have row : edge 2029 = PrimePairCache.smallRow2029.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan185_5 : scan edge (search edge 3) 4 [1123] 1123 innerTail185_5 best = best := by
  rw [innerTail185_5, scan, if_neg (by decide), innerCandidates185_5_checked]
  change scan edge (search edge 3) 4 [1123] 1123 innerTail185_6
    (if 3 ≤ innerCandidates185_5.length then search edge 3 [1123, 2029] innerCandidates185_5 3152 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan185_6

def innerTail185_4 : List Nat := 1783 :: innerTail185_5
def innerCandidates185_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates185_4_checked : candidates edge 4 1123 1783 innerTail185_5 best.bound = innerCandidates185_4 := by
  unfold candidates
  have row : edge 1783 = PrimePairCache.smallRow1783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan185_4 : scan edge (search edge 3) 4 [1123] 1123 innerTail185_4 best = best := by
  rw [innerTail185_4, scan, if_neg (by decide), innerCandidates185_4_checked]
  change scan edge (search edge 3) 4 [1123] 1123 innerTail185_5
    (if 3 ≤ innerCandidates185_4.length then search edge 3 [1123, 1783] innerCandidates185_4 2906 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan185_5

def innerTail185_3 : List Nat := 1657 :: innerTail185_4
def innerCandidates185_3 : List Nat := [2203]
set_option maxRecDepth 4096 in
theorem innerCandidates185_3_checked : candidates edge 4 1123 1657 innerTail185_4 best.bound = innerCandidates185_3 := by
  unfold candidates
  have row : edge 1657 = PrimePairCache.smallRow1657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan185_3 : scan edge (search edge 3) 4 [1123] 1123 innerTail185_3 best = best := by
  rw [innerTail185_3, scan, if_neg (by decide), innerCandidates185_3_checked]
  change scan edge (search edge 3) 4 [1123] 1123 innerTail185_4
    (if 3 ≤ innerCandidates185_3.length then search edge 3 [1123, 1657] innerCandidates185_3 2780 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan185_4

def innerTail185_2 : List Nat := 1453 :: innerTail185_3
def innerCandidates185_2 : List Nat := [5281]
set_option maxRecDepth 4096 in
theorem innerCandidates185_2_checked : candidates edge 4 1123 1453 innerTail185_3 best.bound = innerCandidates185_2 := by
  unfold candidates
  have row : edge 1453 = PrimePairCache.smallRow1453.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan185_2 : scan edge (search edge 3) 4 [1123] 1123 innerTail185_2 best = best := by
  rw [innerTail185_2, scan, if_neg (by decide), innerCandidates185_2_checked]
  change scan edge (search edge 3) 4 [1123] 1123 innerTail185_3
    (if 3 ≤ innerCandidates185_2.length then search edge 3 [1123, 1453] innerCandidates185_2 2576 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan185_3

def innerTail185_1 : List Nat := 1399 :: innerTail185_2
def innerCandidates185_1 : List Nat := [3433]
set_option maxRecDepth 4096 in
theorem innerCandidates185_1_checked : candidates edge 4 1123 1399 innerTail185_2 best.bound = innerCandidates185_1 := by
  unfold candidates
  have row : edge 1399 = PrimePairCache.smallRow1399.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan185_1 : scan edge (search edge 3) 4 [1123] 1123 innerTail185_1 best = best := by
  rw [innerTail185_1, scan, if_neg (by decide), innerCandidates185_1_checked]
  change scan edge (search edge 3) 4 [1123] 1123 innerTail185_2
    (if 3 ≤ innerCandidates185_1.length then search edge 3 [1123, 1399] innerCandidates185_1 2522 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan185_2

def innerTail185_0 : List Nat := 1201 :: innerTail185_1
def innerCandidates185_0 : List Nat := [5281]
set_option maxRecDepth 4096 in
theorem innerCandidates185_0_checked : candidates edge 4 1123 1201 innerTail185_1 best.bound = innerCandidates185_0 := by
  unfold candidates
  have row : edge 1201 = PrimePairCache.smallRow1201.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan185_0 : scan edge (search edge 3) 4 [1123] 1123 innerTail185_0 best = best := by
  rw [innerTail185_0, scan, if_neg (by decide), innerCandidates185_0_checked]
  change scan edge (search edge 3) 4 [1123] 1123 innerTail185_1
    (if 3 ≤ innerCandidates185_0.length then search edge 3 [1123, 1201] innerCandidates185_0 2324 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan185_1

theorem rootBranch185 : search edge 4 [1123] rootCandidates185 1123 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1123] 1123 innerTail185_0 best = best
  exact innerScan185_0

end ProjectEulerSolutions.PrimePairUpper.Search
