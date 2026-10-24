import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates182 : List Nat := [1187, 1373, 1733, 2039, 2081, 2909, 2927, 3041, 3299, 3557, 5237, 5807, 5813, 5987, 6029, 6143, 6317, 6329, 6983, 7529, 8123]
set_option maxRecDepth 4096 in
theorem rootCandidates182_checked : candidates edge 5 0 1103 tail183 best.bound = rootCandidates182 := by
  unfold candidates
  have row : edge 1103 = PrimePairCache.smallRow1103.contains := by funext q; rfl
  rw [row]
  decide

def innerTail182_21 : List Nat := []
theorem innerScan182_21 : scan edge (search edge 3) 4 [1103] 1103 innerTail182_21 best = best := rfl

def innerTail182_20 : List Nat := 8123 :: innerTail182_21
def innerCandidates182_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates182_20_checked : candidates edge 4 1103 8123 innerTail182_21 best.bound = innerCandidates182_20 := by
  unfold candidates
  have row : edge 8123 = SearchData.row8123.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan182_20 : scan edge (search edge 3) 4 [1103] 1103 innerTail182_20 best = best := by
  rw [innerTail182_20, scan, if_neg (by decide), innerCandidates182_20_checked]
  change scan edge (search edge 3) 4 [1103] 1103 innerTail182_21
    (if 3 ≤ innerCandidates182_20.length then search edge 3 [1103, 8123] innerCandidates182_20 9226 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan182_21

def innerTail182_19 : List Nat := 7529 :: innerTail182_20
def innerCandidates182_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates182_19_checked : candidates edge 4 1103 7529 innerTail182_20 best.bound = innerCandidates182_19 := by
  unfold candidates
  have row : edge 7529 = SearchData.row7529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan182_19 : scan edge (search edge 3) 4 [1103] 1103 innerTail182_19 best = best := by
  rw [innerTail182_19, scan, if_neg (by decide), innerCandidates182_19_checked]
  change scan edge (search edge 3) 4 [1103] 1103 innerTail182_20
    (if 3 ≤ innerCandidates182_19.length then search edge 3 [1103, 7529] innerCandidates182_19 8632 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan182_20

def innerTail182_18 : List Nat := 6983 :: innerTail182_19
def innerCandidates182_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates182_18_checked : candidates edge 4 1103 6983 innerTail182_19 best.bound = innerCandidates182_18 := by
  unfold candidates
  have row : edge 6983 = SearchData.row6983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan182_18 : scan edge (search edge 3) 4 [1103] 1103 innerTail182_18 best = best := by
  rw [innerTail182_18, scan, if_neg (by decide), innerCandidates182_18_checked]
  change scan edge (search edge 3) 4 [1103] 1103 innerTail182_19
    (if 3 ≤ innerCandidates182_18.length then search edge 3 [1103, 6983] innerCandidates182_18 8086 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan182_19

def innerTail182_17 : List Nat := 6329 :: innerTail182_18
def innerCandidates182_17 : List Nat := [6983]
set_option maxRecDepth 4096 in
theorem innerCandidates182_17_checked : candidates edge 4 1103 6329 innerTail182_18 best.bound = innerCandidates182_17 := by
  unfold candidates
  have row : edge 6329 = SearchData.row6329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan182_17 : scan edge (search edge 3) 4 [1103] 1103 innerTail182_17 best = best := by
  rw [innerTail182_17, scan, if_neg (by decide), innerCandidates182_17_checked]
  change scan edge (search edge 3) 4 [1103] 1103 innerTail182_18
    (if 3 ≤ innerCandidates182_17.length then search edge 3 [1103, 6329] innerCandidates182_17 7432 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan182_18

def innerTail182_16 : List Nat := 6317 :: innerTail182_17
def innerCandidates182_16 : List Nat := [6983, 7529]
set_option maxRecDepth 4096 in
theorem innerCandidates182_16_checked : candidates edge 4 1103 6317 innerTail182_17 best.bound = innerCandidates182_16 := by
  unfold candidates
  have row : edge 6317 = SearchData.row6317.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan182_16 : scan edge (search edge 3) 4 [1103] 1103 innerTail182_16 best = best := by
  rw [innerTail182_16, scan, if_neg (by decide), innerCandidates182_16_checked]
  change scan edge (search edge 3) 4 [1103] 1103 innerTail182_17
    (if 3 ≤ innerCandidates182_16.length then search edge 3 [1103, 6317] innerCandidates182_16 7420 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan182_17

def innerTail182_15 : List Nat := 6143 :: innerTail182_16
def innerCandidates182_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates182_15_checked : candidates edge 4 1103 6143 innerTail182_16 best.bound = innerCandidates182_15 := by
  unfold candidates
  have row : edge 6143 = SearchData.row6143.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan182_15 : scan edge (search edge 3) 4 [1103] 1103 innerTail182_15 best = best := by
  rw [innerTail182_15, scan, if_neg (by decide), innerCandidates182_15_checked]
  change scan edge (search edge 3) 4 [1103] 1103 innerTail182_16
    (if 3 ≤ innerCandidates182_15.length then search edge 3 [1103, 6143] innerCandidates182_15 7246 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan182_16

def innerTail182_14 : List Nat := 6029 :: innerTail182_15
def innerCandidates182_14 : List Nat := [7529]
set_option maxRecDepth 4096 in
theorem innerCandidates182_14_checked : candidates edge 4 1103 6029 innerTail182_15 best.bound = innerCandidates182_14 := by
  unfold candidates
  have row : edge 6029 = SearchData.row6029.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan182_14 : scan edge (search edge 3) 4 [1103] 1103 innerTail182_14 best = best := by
  rw [innerTail182_14, scan, if_neg (by decide), innerCandidates182_14_checked]
  change scan edge (search edge 3) 4 [1103] 1103 innerTail182_15
    (if 3 ≤ innerCandidates182_14.length then search edge 3 [1103, 6029] innerCandidates182_14 7132 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan182_15

def innerTail182_13 : List Nat := 5987 :: innerTail182_14
def innerCandidates182_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates182_13_checked : candidates edge 4 1103 5987 innerTail182_14 best.bound = innerCandidates182_13 := by
  unfold candidates
  have row : edge 5987 = SearchData.row5987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan182_13 : scan edge (search edge 3) 4 [1103] 1103 innerTail182_13 best = best := by
  rw [innerTail182_13, scan, if_neg (by decide), innerCandidates182_13_checked]
  change scan edge (search edge 3) 4 [1103] 1103 innerTail182_14
    (if 3 ≤ innerCandidates182_13.length then search edge 3 [1103, 5987] innerCandidates182_13 7090 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan182_14

def innerTail182_12 : List Nat := 5813 :: innerTail182_13
def innerCandidates182_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates182_12_checked : candidates edge 4 1103 5813 innerTail182_13 best.bound = innerCandidates182_12 := by
  unfold candidates
  have row : edge 5813 = SearchData.row5813.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan182_12 : scan edge (search edge 3) 4 [1103] 1103 innerTail182_12 best = best := by
  rw [innerTail182_12, scan, if_neg (by decide), innerCandidates182_12_checked]
  change scan edge (search edge 3) 4 [1103] 1103 innerTail182_13
    (if 3 ≤ innerCandidates182_12.length then search edge 3 [1103, 5813] innerCandidates182_12 6916 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan182_13

def innerTail182_11 : List Nat := 5807 :: innerTail182_12
def innerCandidates182_11 : List Nat := [6329]
set_option maxRecDepth 4096 in
theorem innerCandidates182_11_checked : candidates edge 4 1103 5807 innerTail182_12 best.bound = innerCandidates182_11 := by
  unfold candidates
  have row : edge 5807 = SearchData.row5807.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan182_11 : scan edge (search edge 3) 4 [1103] 1103 innerTail182_11 best = best := by
  rw [innerTail182_11, scan, if_neg (by decide), innerCandidates182_11_checked]
  change scan edge (search edge 3) 4 [1103] 1103 innerTail182_12
    (if 3 ≤ innerCandidates182_11.length then search edge 3 [1103, 5807] innerCandidates182_11 6910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan182_12

def innerTail182_10 : List Nat := 5237 :: innerTail182_11
def innerCandidates182_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates182_10_checked : candidates edge 4 1103 5237 innerTail182_11 best.bound = innerCandidates182_10 := by
  unfold candidates
  have row : edge 5237 = SearchData.row5237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan182_10 : scan edge (search edge 3) 4 [1103] 1103 innerTail182_10 best = best := by
  rw [innerTail182_10, scan, if_neg (by decide), innerCandidates182_10_checked]
  change scan edge (search edge 3) 4 [1103] 1103 innerTail182_11
    (if 3 ≤ innerCandidates182_10.length then search edge 3 [1103, 5237] innerCandidates182_10 6340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan182_11

def innerTail182_9 : List Nat := 3557 :: innerTail182_10
def innerCandidates182_9 : List Nat := [6329]
set_option maxRecDepth 4096 in
theorem innerCandidates182_9_checked : candidates edge 4 1103 3557 innerTail182_10 best.bound = innerCandidates182_9 := by
  unfold candidates
  have row : edge 3557 = SearchData.row3557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan182_9 : scan edge (search edge 3) 4 [1103] 1103 innerTail182_9 best = best := by
  rw [innerTail182_9, scan, if_neg (by decide), innerCandidates182_9_checked]
  change scan edge (search edge 3) 4 [1103] 1103 innerTail182_10
    (if 3 ≤ innerCandidates182_9.length then search edge 3 [1103, 3557] innerCandidates182_9 4660 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan182_10

def innerTail182_8 : List Nat := 3299 :: innerTail182_9
def innerCandidates182_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates182_8_checked : candidates edge 4 1103 3299 innerTail182_9 best.bound = innerCandidates182_8 := by
  unfold candidates
  have row : edge 3299 = SearchData.row3299.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan182_8 : scan edge (search edge 3) 4 [1103] 1103 innerTail182_8 best = best := by
  rw [innerTail182_8, scan, if_neg (by decide), innerCandidates182_8_checked]
  change scan edge (search edge 3) 4 [1103] 1103 innerTail182_9
    (if 3 ≤ innerCandidates182_8.length then search edge 3 [1103, 3299] innerCandidates182_8 4402 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan182_9

def innerTail182_7 : List Nat := 3041 :: innerTail182_8
def innerCandidates182_7 : List Nat := [6317, 7529]
set_option maxRecDepth 4096 in
theorem innerCandidates182_7_checked : candidates edge 4 1103 3041 innerTail182_8 best.bound = innerCandidates182_7 := by
  unfold candidates
  have row : edge 3041 = SearchData.row3041.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan182_7 : scan edge (search edge 3) 4 [1103] 1103 innerTail182_7 best = best := by
  rw [innerTail182_7, scan, if_neg (by decide), innerCandidates182_7_checked]
  change scan edge (search edge 3) 4 [1103] 1103 innerTail182_8
    (if 3 ≤ innerCandidates182_7.length then search edge 3 [1103, 3041] innerCandidates182_7 4144 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan182_8

def innerTail182_6 : List Nat := 2927 :: innerTail182_7
def innerCandidates182_6 : List Nat := [6029, 6983]
set_option maxRecDepth 4096 in
theorem innerCandidates182_6_checked : candidates edge 4 1103 2927 innerTail182_7 best.bound = innerCandidates182_6 := by
  unfold candidates
  have row : edge 2927 = SearchData.row2927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan182_6 : scan edge (search edge 3) 4 [1103] 1103 innerTail182_6 best = best := by
  rw [innerTail182_6, scan, if_neg (by decide), innerCandidates182_6_checked]
  change scan edge (search edge 3) 4 [1103] 1103 innerTail182_7
    (if 3 ≤ innerCandidates182_6.length then search edge 3 [1103, 2927] innerCandidates182_6 4030 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan182_7

def innerTail182_5 : List Nat := 2909 :: innerTail182_6
def innerCandidates182_5 : List Nat := [3557, 5813, 6983]
set_option maxRecDepth 4096 in
theorem innerCandidates182_5_checked : candidates edge 4 1103 2909 innerTail182_6 best.bound = innerCandidates182_5 := by
  unfold candidates
  have row : edge 2909 = SearchData.row2909.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild182_5 : search edge 3 [1103, 2909] innerCandidates182_5 4012 best = best := by decide

theorem innerScan182_5 : scan edge (search edge 3) 4 [1103] 1103 innerTail182_5 best = best := by
  rw [innerTail182_5, scan, if_neg (by decide), innerCandidates182_5_checked]
  change scan edge (search edge 3) 4 [1103] 1103 innerTail182_6
    (if 3 ≤ innerCandidates182_5.length then search edge 3 [1103, 2909] innerCandidates182_5 4012 best else best) = best
  rw [if_pos (by decide), innerChild182_5]
  exact innerScan182_6

def innerTail182_4 : List Nat := 2081 :: innerTail182_5
def innerCandidates182_4 : List Nat := [5807]
set_option maxRecDepth 4096 in
theorem innerCandidates182_4_checked : candidates edge 4 1103 2081 innerTail182_5 best.bound = innerCandidates182_4 := by
  unfold candidates
  have row : edge 2081 = PrimePairCache.smallRow2081.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan182_4 : scan edge (search edge 3) 4 [1103] 1103 innerTail182_4 best = best := by
  rw [innerTail182_4, scan, if_neg (by decide), innerCandidates182_4_checked]
  change scan edge (search edge 3) 4 [1103] 1103 innerTail182_5
    (if 3 ≤ innerCandidates182_4.length then search edge 3 [1103, 2081] innerCandidates182_4 3184 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan182_5

def innerTail182_3 : List Nat := 2039 :: innerTail182_4
def innerCandidates182_3 : List Nat := [3299]
set_option maxRecDepth 4096 in
theorem innerCandidates182_3_checked : candidates edge 4 1103 2039 innerTail182_4 best.bound = innerCandidates182_3 := by
  unfold candidates
  have row : edge 2039 = PrimePairCache.smallRow2039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan182_3 : scan edge (search edge 3) 4 [1103] 1103 innerTail182_3 best = best := by
  rw [innerTail182_3, scan, if_neg (by decide), innerCandidates182_3_checked]
  change scan edge (search edge 3) 4 [1103] 1103 innerTail182_4
    (if 3 ≤ innerCandidates182_3.length then search edge 3 [1103, 2039] innerCandidates182_3 3142 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan182_4

def innerTail182_2 : List Nat := 1733 :: innerTail182_3
def innerCandidates182_2 : List Nat := [2927]
set_option maxRecDepth 4096 in
theorem innerCandidates182_2_checked : candidates edge 4 1103 1733 innerTail182_3 best.bound = innerCandidates182_2 := by
  unfold candidates
  have row : edge 1733 = PrimePairCache.smallRow1733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan182_2 : scan edge (search edge 3) 4 [1103] 1103 innerTail182_2 best = best := by
  rw [innerTail182_2, scan, if_neg (by decide), innerCandidates182_2_checked]
  change scan edge (search edge 3) 4 [1103] 1103 innerTail182_3
    (if 3 ≤ innerCandidates182_2.length then search edge 3 [1103, 1733] innerCandidates182_2 2836 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan182_3

def innerTail182_1 : List Nat := 1373 :: innerTail182_2
def innerCandidates182_1 : List Nat := [6143]
set_option maxRecDepth 4096 in
theorem innerCandidates182_1_checked : candidates edge 4 1103 1373 innerTail182_2 best.bound = innerCandidates182_1 := by
  unfold candidates
  have row : edge 1373 = PrimePairCache.smallRow1373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan182_1 : scan edge (search edge 3) 4 [1103] 1103 innerTail182_1 best = best := by
  rw [innerTail182_1, scan, if_neg (by decide), innerCandidates182_1_checked]
  change scan edge (search edge 3) 4 [1103] 1103 innerTail182_2
    (if 3 ≤ innerCandidates182_1.length then search edge 3 [1103, 1373] innerCandidates182_1 2476 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan182_2

def innerTail182_0 : List Nat := 1187 :: innerTail182_1
def innerCandidates182_0 : List Nat := [6983]
set_option maxRecDepth 4096 in
theorem innerCandidates182_0_checked : candidates edge 4 1103 1187 innerTail182_1 best.bound = innerCandidates182_0 := by
  unfold candidates
  have row : edge 1187 = PrimePairCache.smallRow1187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan182_0 : scan edge (search edge 3) 4 [1103] 1103 innerTail182_0 best = best := by
  rw [innerTail182_0, scan, if_neg (by decide), innerCandidates182_0_checked]
  change scan edge (search edge 3) 4 [1103] 1103 innerTail182_1
    (if 3 ≤ innerCandidates182_0.length then search edge 3 [1103, 1187] innerCandidates182_0 2290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan182_1

theorem rootBranch182 : search edge 4 [1103] rootCandidates182 1103 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1103] 1103 innerTail182_0 best = best
  exact innerScan182_0

def rootCandidates181 : List Nat := [1637, 2393, 2999, 3251, 3533, 3593, 4283, 4409, 4493, 4637, 4751, 5087, 5651, 6113, 6323, 6473, 7517, 7559, 7727, 8087]
set_option maxRecDepth 4096 in
theorem rootCandidates181_checked : candidates edge 5 0 1097 tail182 best.bound = rootCandidates181 := by
  unfold candidates
  have row : edge 1097 = PrimePairCache.smallRow1097.contains := by funext q; rfl
  rw [row]
  decide

def innerTail181_20 : List Nat := []
theorem innerScan181_20 : scan edge (search edge 3) 4 [1097] 1097 innerTail181_20 best = best := rfl

def innerTail181_19 : List Nat := 8087 :: innerTail181_20
def innerCandidates181_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates181_19_checked : candidates edge 4 1097 8087 innerTail181_20 best.bound = innerCandidates181_19 := by
  unfold candidates
  have row : edge 8087 = SearchData.row8087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan181_19 : scan edge (search edge 3) 4 [1097] 1097 innerTail181_19 best = best := by
  rw [innerTail181_19, scan, if_neg (by decide), innerCandidates181_19_checked]
  change scan edge (search edge 3) 4 [1097] 1097 innerTail181_20
    (if 3 ≤ innerCandidates181_19.length then search edge 3 [1097, 8087] innerCandidates181_19 9184 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan181_20

def innerTail181_18 : List Nat := 7727 :: innerTail181_19
def innerCandidates181_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates181_18_checked : candidates edge 4 1097 7727 innerTail181_19 best.bound = innerCandidates181_18 := by
  unfold candidates
  have row : edge 7727 = SearchData.row7727.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan181_18 : scan edge (search edge 3) 4 [1097] 1097 innerTail181_18 best = best := by
  rw [innerTail181_18, scan, if_neg (by decide), innerCandidates181_18_checked]
  change scan edge (search edge 3) 4 [1097] 1097 innerTail181_19
    (if 3 ≤ innerCandidates181_18.length then search edge 3 [1097, 7727] innerCandidates181_18 8824 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan181_19

def innerTail181_17 : List Nat := 7559 :: innerTail181_18
def innerCandidates181_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates181_17_checked : candidates edge 4 1097 7559 innerTail181_18 best.bound = innerCandidates181_17 := by
  unfold candidates
  have row : edge 7559 = SearchData.row7559.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan181_17 : scan edge (search edge 3) 4 [1097] 1097 innerTail181_17 best = best := by
  rw [innerTail181_17, scan, if_neg (by decide), innerCandidates181_17_checked]
  change scan edge (search edge 3) 4 [1097] 1097 innerTail181_18
    (if 3 ≤ innerCandidates181_17.length then search edge 3 [1097, 7559] innerCandidates181_17 8656 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan181_18

def innerTail181_16 : List Nat := 7517 :: innerTail181_17
def innerCandidates181_16 : List Nat := [7727]
set_option maxRecDepth 4096 in
theorem innerCandidates181_16_checked : candidates edge 4 1097 7517 innerTail181_17 best.bound = innerCandidates181_16 := by
  unfold candidates
  have row : edge 7517 = SearchData.row7517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan181_16 : scan edge (search edge 3) 4 [1097] 1097 innerTail181_16 best = best := by
  rw [innerTail181_16, scan, if_neg (by decide), innerCandidates181_16_checked]
  change scan edge (search edge 3) 4 [1097] 1097 innerTail181_17
    (if 3 ≤ innerCandidates181_16.length then search edge 3 [1097, 7517] innerCandidates181_16 8614 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan181_17

def innerTail181_15 : List Nat := 6473 :: innerTail181_16
def innerCandidates181_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates181_15_checked : candidates edge 4 1097 6473 innerTail181_16 best.bound = innerCandidates181_15 := by
  unfold candidates
  have row : edge 6473 = SearchData.row6473.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan181_15 : scan edge (search edge 3) 4 [1097] 1097 innerTail181_15 best = best := by
  rw [innerTail181_15, scan, if_neg (by decide), innerCandidates181_15_checked]
  change scan edge (search edge 3) 4 [1097] 1097 innerTail181_16
    (if 3 ≤ innerCandidates181_15.length then search edge 3 [1097, 6473] innerCandidates181_15 7570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan181_16

def innerTail181_14 : List Nat := 6323 :: innerTail181_15
def innerCandidates181_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates181_14_checked : candidates edge 4 1097 6323 innerTail181_15 best.bound = innerCandidates181_14 := by
  unfold candidates
  have row : edge 6323 = SearchData.row6323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan181_14 : scan edge (search edge 3) 4 [1097] 1097 innerTail181_14 best = best := by
  rw [innerTail181_14, scan, if_neg (by decide), innerCandidates181_14_checked]
  change scan edge (search edge 3) 4 [1097] 1097 innerTail181_15
    (if 3 ≤ innerCandidates181_14.length then search edge 3 [1097, 6323] innerCandidates181_14 7420 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan181_15

def innerTail181_13 : List Nat := 6113 :: innerTail181_14
def innerCandidates181_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates181_13_checked : candidates edge 4 1097 6113 innerTail181_14 best.bound = innerCandidates181_13 := by
  unfold candidates
  have row : edge 6113 = SearchData.row6113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan181_13 : scan edge (search edge 3) 4 [1097] 1097 innerTail181_13 best = best := by
  rw [innerTail181_13, scan, if_neg (by decide), innerCandidates181_13_checked]
  change scan edge (search edge 3) 4 [1097] 1097 innerTail181_14
    (if 3 ≤ innerCandidates181_13.length then search edge 3 [1097, 6113] innerCandidates181_13 7210 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan181_14

def innerTail181_12 : List Nat := 5651 :: innerTail181_13
def innerCandidates181_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates181_12_checked : candidates edge 4 1097 5651 innerTail181_13 best.bound = innerCandidates181_12 := by
  unfold candidates
  have row : edge 5651 = SearchData.row5651.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan181_12 : scan edge (search edge 3) 4 [1097] 1097 innerTail181_12 best = best := by
  rw [innerTail181_12, scan, if_neg (by decide), innerCandidates181_12_checked]
  change scan edge (search edge 3) 4 [1097] 1097 innerTail181_13
    (if 3 ≤ innerCandidates181_12.length then search edge 3 [1097, 5651] innerCandidates181_12 6748 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan181_13

def innerTail181_11 : List Nat := 5087 :: innerTail181_12
def innerCandidates181_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates181_11_checked : candidates edge 4 1097 5087 innerTail181_12 best.bound = innerCandidates181_11 := by
  unfold candidates
  have row : edge 5087 = SearchData.row5087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan181_11 : scan edge (search edge 3) 4 [1097] 1097 innerTail181_11 best = best := by
  rw [innerTail181_11, scan, if_neg (by decide), innerCandidates181_11_checked]
  change scan edge (search edge 3) 4 [1097] 1097 innerTail181_12
    (if 3 ≤ innerCandidates181_11.length then search edge 3 [1097, 5087] innerCandidates181_11 6184 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan181_12

def innerTail181_10 : List Nat := 4751 :: innerTail181_11
def innerCandidates181_10 : List Nat := [6113]
set_option maxRecDepth 4096 in
theorem innerCandidates181_10_checked : candidates edge 4 1097 4751 innerTail181_11 best.bound = innerCandidates181_10 := by
  unfold candidates
  have row : edge 4751 = SearchData.row4751.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan181_10 : scan edge (search edge 3) 4 [1097] 1097 innerTail181_10 best = best := by
  rw [innerTail181_10, scan, if_neg (by decide), innerCandidates181_10_checked]
  change scan edge (search edge 3) 4 [1097] 1097 innerTail181_11
    (if 3 ≤ innerCandidates181_10.length then search edge 3 [1097, 4751] innerCandidates181_10 5848 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan181_11

def innerTail181_9 : List Nat := 4637 :: innerTail181_10
def innerCandidates181_9 : List Nat := [7517]
set_option maxRecDepth 4096 in
theorem innerCandidates181_9_checked : candidates edge 4 1097 4637 innerTail181_10 best.bound = innerCandidates181_9 := by
  unfold candidates
  have row : edge 4637 = SearchData.row4637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan181_9 : scan edge (search edge 3) 4 [1097] 1097 innerTail181_9 best = best := by
  rw [innerTail181_9, scan, if_neg (by decide), innerCandidates181_9_checked]
  change scan edge (search edge 3) 4 [1097] 1097 innerTail181_10
    (if 3 ≤ innerCandidates181_9.length then search edge 3 [1097, 4637] innerCandidates181_9 5734 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan181_10

def innerTail181_8 : List Nat := 4493 :: innerTail181_9
def innerCandidates181_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates181_8_checked : candidates edge 4 1097 4493 innerTail181_9 best.bound = innerCandidates181_8 := by
  unfold candidates
  have row : edge 4493 = SearchData.row4493.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan181_8 : scan edge (search edge 3) 4 [1097] 1097 innerTail181_8 best = best := by
  rw [innerTail181_8, scan, if_neg (by decide), innerCandidates181_8_checked]
  change scan edge (search edge 3) 4 [1097] 1097 innerTail181_9
    (if 3 ≤ innerCandidates181_8.length then search edge 3 [1097, 4493] innerCandidates181_8 5590 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan181_9

def innerTail181_7 : List Nat := 4409 :: innerTail181_8
def innerCandidates181_7 : List Nat := [5087]
set_option maxRecDepth 4096 in
theorem innerCandidates181_7_checked : candidates edge 4 1097 4409 innerTail181_8 best.bound = innerCandidates181_7 := by
  unfold candidates
  have row : edge 4409 = SearchData.row4409.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan181_7 : scan edge (search edge 3) 4 [1097] 1097 innerTail181_7 best = best := by
  rw [innerTail181_7, scan, if_neg (by decide), innerCandidates181_7_checked]
  change scan edge (search edge 3) 4 [1097] 1097 innerTail181_8
    (if 3 ≤ innerCandidates181_7.length then search edge 3 [1097, 4409] innerCandidates181_7 5506 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan181_8

def innerTail181_6 : List Nat := 4283 :: innerTail181_7
def innerCandidates181_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates181_6_checked : candidates edge 4 1097 4283 innerTail181_7 best.bound = innerCandidates181_6 := by
  unfold candidates
  have row : edge 4283 = SearchData.row4283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan181_6 : scan edge (search edge 3) 4 [1097] 1097 innerTail181_6 best = best := by
  rw [innerTail181_6, scan, if_neg (by decide), innerCandidates181_6_checked]
  change scan edge (search edge 3) 4 [1097] 1097 innerTail181_7
    (if 3 ≤ innerCandidates181_6.length then search edge 3 [1097, 4283] innerCandidates181_6 5380 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan181_7

def innerTail181_5 : List Nat := 3593 :: innerTail181_6
def innerCandidates181_5 : List Nat := [5087]
set_option maxRecDepth 4096 in
theorem innerCandidates181_5_checked : candidates edge 4 1097 3593 innerTail181_6 best.bound = innerCandidates181_5 := by
  unfold candidates
  have row : edge 3593 = SearchData.row3593.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan181_5 : scan edge (search edge 3) 4 [1097] 1097 innerTail181_5 best = best := by
  rw [innerTail181_5, scan, if_neg (by decide), innerCandidates181_5_checked]
  change scan edge (search edge 3) 4 [1097] 1097 innerTail181_6
    (if 3 ≤ innerCandidates181_5.length then search edge 3 [1097, 3593] innerCandidates181_5 4690 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan181_6

def innerTail181_4 : List Nat := 3533 :: innerTail181_5
def innerCandidates181_4 : List Nat := [3593, 6113]
set_option maxRecDepth 4096 in
theorem innerCandidates181_4_checked : candidates edge 4 1097 3533 innerTail181_5 best.bound = innerCandidates181_4 := by
  unfold candidates
  have row : edge 3533 = SearchData.row3533.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan181_4 : scan edge (search edge 3) 4 [1097] 1097 innerTail181_4 best = best := by
  rw [innerTail181_4, scan, if_neg (by decide), innerCandidates181_4_checked]
  change scan edge (search edge 3) 4 [1097] 1097 innerTail181_5
    (if 3 ≤ innerCandidates181_4.length then search edge 3 [1097, 3533] innerCandidates181_4 4630 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan181_5

def innerTail181_3 : List Nat := 3251 :: innerTail181_4
def innerCandidates181_3 : List Nat := [4409]
set_option maxRecDepth 4096 in
theorem innerCandidates181_3_checked : candidates edge 4 1097 3251 innerTail181_4 best.bound = innerCandidates181_3 := by
  unfold candidates
  have row : edge 3251 = SearchData.row3251.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan181_3 : scan edge (search edge 3) 4 [1097] 1097 innerTail181_3 best = best := by
  rw [innerTail181_3, scan, if_neg (by decide), innerCandidates181_3_checked]
  change scan edge (search edge 3) 4 [1097] 1097 innerTail181_4
    (if 3 ≤ innerCandidates181_3.length then search edge 3 [1097, 3251] innerCandidates181_3 4348 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan181_4

def innerTail181_2 : List Nat := 2999 :: innerTail181_3
def innerCandidates181_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates181_2_checked : candidates edge 4 1097 2999 innerTail181_3 best.bound = innerCandidates181_2 := by
  unfold candidates
  have row : edge 2999 = SearchData.row2999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan181_2 : scan edge (search edge 3) 4 [1097] 1097 innerTail181_2 best = best := by
  rw [innerTail181_2, scan, if_neg (by decide), innerCandidates181_2_checked]
  change scan edge (search edge 3) 4 [1097] 1097 innerTail181_3
    (if 3 ≤ innerCandidates181_2.length then search edge 3 [1097, 2999] innerCandidates181_2 4096 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan181_3

def innerTail181_1 : List Nat := 2393 :: innerTail181_2
def innerCandidates181_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates181_1_checked : candidates edge 4 1097 2393 innerTail181_2 best.bound = innerCandidates181_1 := by
  unfold candidates
  have row : edge 2393 = PrimePairCache.smallRow2393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan181_1 : scan edge (search edge 3) 4 [1097] 1097 innerTail181_1 best = best := by
  rw [innerTail181_1, scan, if_neg (by decide), innerCandidates181_1_checked]
  change scan edge (search edge 3) 4 [1097] 1097 innerTail181_2
    (if 3 ≤ innerCandidates181_1.length then search edge 3 [1097, 2393] innerCandidates181_1 3490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan181_2

def innerTail181_0 : List Nat := 1637 :: innerTail181_1
def innerCandidates181_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates181_0_checked : candidates edge 4 1097 1637 innerTail181_1 best.bound = innerCandidates181_0 := by
  unfold candidates
  have row : edge 1637 = PrimePairCache.smallRow1637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan181_0 : scan edge (search edge 3) 4 [1097] 1097 innerTail181_0 best = best := by
  rw [innerTail181_0, scan, if_neg (by decide), innerCandidates181_0_checked]
  change scan edge (search edge 3) 4 [1097] 1097 innerTail181_1
    (if 3 ≤ innerCandidates181_0.length then search edge 3 [1097, 1637] innerCandidates181_0 2734 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan181_1

theorem rootBranch181 : search edge 4 [1097] rootCandidates181 1097 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1097] 1097 innerTail181_0 best = best
  exact innerScan181_0

def rootCandidates180 : List Nat := [1471, 1549, 1861, 2437, 3061, 3331, 4201, 4243, 4447, 4603, 4813, 5413, 5569, 5779, 6427, 6481, 6553, 6619, 7057, 7309, 7741]
set_option maxRecDepth 4096 in
theorem rootCandidates180_checked : candidates edge 5 0 1093 tail181 best.bound = rootCandidates180 := by
  unfold candidates
  have row : edge 1093 = PrimePairCache.smallRow1093.contains := by funext q; rfl
  rw [row]
  decide

def innerTail180_21 : List Nat := []
theorem innerScan180_21 : scan edge (search edge 3) 4 [1093] 1093 innerTail180_21 best = best := rfl

def innerTail180_20 : List Nat := 7741 :: innerTail180_21
def innerCandidates180_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates180_20_checked : candidates edge 4 1093 7741 innerTail180_21 best.bound = innerCandidates180_20 := by
  unfold candidates
  have row : edge 7741 = SearchData.row7741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan180_20 : scan edge (search edge 3) 4 [1093] 1093 innerTail180_20 best = best := by
  rw [innerTail180_20, scan, if_neg (by decide), innerCandidates180_20_checked]
  change scan edge (search edge 3) 4 [1093] 1093 innerTail180_21
    (if 3 ≤ innerCandidates180_20.length then search edge 3 [1093, 7741] innerCandidates180_20 8834 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan180_21

def innerTail180_19 : List Nat := 7309 :: innerTail180_20
def innerCandidates180_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates180_19_checked : candidates edge 4 1093 7309 innerTail180_20 best.bound = innerCandidates180_19 := by
  unfold candidates
  have row : edge 7309 = SearchData.row7309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan180_19 : scan edge (search edge 3) 4 [1093] 1093 innerTail180_19 best = best := by
  rw [innerTail180_19, scan, if_neg (by decide), innerCandidates180_19_checked]
  change scan edge (search edge 3) 4 [1093] 1093 innerTail180_20
    (if 3 ≤ innerCandidates180_19.length then search edge 3 [1093, 7309] innerCandidates180_19 8402 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan180_20

def innerTail180_18 : List Nat := 7057 :: innerTail180_19
def innerCandidates180_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates180_18_checked : candidates edge 4 1093 7057 innerTail180_19 best.bound = innerCandidates180_18 := by
  unfold candidates
  have row : edge 7057 = SearchData.row7057.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan180_18 : scan edge (search edge 3) 4 [1093] 1093 innerTail180_18 best = best := by
  rw [innerTail180_18, scan, if_neg (by decide), innerCandidates180_18_checked]
  change scan edge (search edge 3) 4 [1093] 1093 innerTail180_19
    (if 3 ≤ innerCandidates180_18.length then search edge 3 [1093, 7057] innerCandidates180_18 8150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan180_19

def innerTail180_17 : List Nat := 6619 :: innerTail180_18
def innerCandidates180_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates180_17_checked : candidates edge 4 1093 6619 innerTail180_18 best.bound = innerCandidates180_17 := by
  unfold candidates
  have row : edge 6619 = SearchData.row6619.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan180_17 : scan edge (search edge 3) 4 [1093] 1093 innerTail180_17 best = best := by
  rw [innerTail180_17, scan, if_neg (by decide), innerCandidates180_17_checked]
  change scan edge (search edge 3) 4 [1093] 1093 innerTail180_18
    (if 3 ≤ innerCandidates180_17.length then search edge 3 [1093, 6619] innerCandidates180_17 7712 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan180_18

def innerTail180_16 : List Nat := 6553 :: innerTail180_17
def innerCandidates180_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates180_16_checked : candidates edge 4 1093 6553 innerTail180_17 best.bound = innerCandidates180_16 := by
  unfold candidates
  have row : edge 6553 = SearchData.row6553.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan180_16 : scan edge (search edge 3) 4 [1093] 1093 innerTail180_16 best = best := by
  rw [innerTail180_16, scan, if_neg (by decide), innerCandidates180_16_checked]
  change scan edge (search edge 3) 4 [1093] 1093 innerTail180_17
    (if 3 ≤ innerCandidates180_16.length then search edge 3 [1093, 6553] innerCandidates180_16 7646 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan180_17

def innerTail180_15 : List Nat := 6481 :: innerTail180_16
def innerCandidates180_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates180_15_checked : candidates edge 4 1093 6481 innerTail180_16 best.bound = innerCandidates180_15 := by
  unfold candidates
  have row : edge 6481 = SearchData.row6481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan180_15 : scan edge (search edge 3) 4 [1093] 1093 innerTail180_15 best = best := by
  rw [innerTail180_15, scan, if_neg (by decide), innerCandidates180_15_checked]
  change scan edge (search edge 3) 4 [1093] 1093 innerTail180_16
    (if 3 ≤ innerCandidates180_15.length then search edge 3 [1093, 6481] innerCandidates180_15 7574 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan180_16

def innerTail180_14 : List Nat := 6427 :: innerTail180_15
def innerCandidates180_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates180_14_checked : candidates edge 4 1093 6427 innerTail180_15 best.bound = innerCandidates180_14 := by
  unfold candidates
  have row : edge 6427 = SearchData.row6427.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan180_14 : scan edge (search edge 3) 4 [1093] 1093 innerTail180_14 best = best := by
  rw [innerTail180_14, scan, if_neg (by decide), innerCandidates180_14_checked]
  change scan edge (search edge 3) 4 [1093] 1093 innerTail180_15
    (if 3 ≤ innerCandidates180_14.length then search edge 3 [1093, 6427] innerCandidates180_14 7520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan180_15

def innerTail180_13 : List Nat := 5779 :: innerTail180_14
def innerCandidates180_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates180_13_checked : candidates edge 4 1093 5779 innerTail180_14 best.bound = innerCandidates180_13 := by
  unfold candidates
  have row : edge 5779 = SearchData.row5779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan180_13 : scan edge (search edge 3) 4 [1093] 1093 innerTail180_13 best = best := by
  rw [innerTail180_13, scan, if_neg (by decide), innerCandidates180_13_checked]
  change scan edge (search edge 3) 4 [1093] 1093 innerTail180_14
    (if 3 ≤ innerCandidates180_13.length then search edge 3 [1093, 5779] innerCandidates180_13 6872 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan180_14

def innerTail180_12 : List Nat := 5569 :: innerTail180_13
def innerCandidates180_12 : List Nat := [7309]
set_option maxRecDepth 4096 in
theorem innerCandidates180_12_checked : candidates edge 4 1093 5569 innerTail180_13 best.bound = innerCandidates180_12 := by
  unfold candidates
  have row : edge 5569 = SearchData.row5569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan180_12 : scan edge (search edge 3) 4 [1093] 1093 innerTail180_12 best = best := by
  rw [innerTail180_12, scan, if_neg (by decide), innerCandidates180_12_checked]
  change scan edge (search edge 3) 4 [1093] 1093 innerTail180_13
    (if 3 ≤ innerCandidates180_12.length then search edge 3 [1093, 5569] innerCandidates180_12 6662 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan180_13

def innerTail180_11 : List Nat := 5413 :: innerTail180_12
def innerCandidates180_11 : List Nat := [5569]
set_option maxRecDepth 4096 in
theorem innerCandidates180_11_checked : candidates edge 4 1093 5413 innerTail180_12 best.bound = innerCandidates180_11 := by
  unfold candidates
  have row : edge 5413 = SearchData.row5413.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan180_11 : scan edge (search edge 3) 4 [1093] 1093 innerTail180_11 best = best := by
  rw [innerTail180_11, scan, if_neg (by decide), innerCandidates180_11_checked]
  change scan edge (search edge 3) 4 [1093] 1093 innerTail180_12
    (if 3 ≤ innerCandidates180_11.length then search edge 3 [1093, 5413] innerCandidates180_11 6506 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan180_12

def innerTail180_10 : List Nat := 4813 :: innerTail180_11
def innerCandidates180_10 : List Nat := [6427, 7057]
set_option maxRecDepth 4096 in
theorem innerCandidates180_10_checked : candidates edge 4 1093 4813 innerTail180_11 best.bound = innerCandidates180_10 := by
  unfold candidates
  have row : edge 4813 = SearchData.row4813.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan180_10 : scan edge (search edge 3) 4 [1093] 1093 innerTail180_10 best = best := by
  rw [innerTail180_10, scan, if_neg (by decide), innerCandidates180_10_checked]
  change scan edge (search edge 3) 4 [1093] 1093 innerTail180_11
    (if 3 ≤ innerCandidates180_10.length then search edge 3 [1093, 4813] innerCandidates180_10 5906 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan180_11

def innerTail180_9 : List Nat := 4603 :: innerTail180_10
def innerCandidates180_9 : List Nat := [5569]
set_option maxRecDepth 4096 in
theorem innerCandidates180_9_checked : candidates edge 4 1093 4603 innerTail180_10 best.bound = innerCandidates180_9 := by
  unfold candidates
  have row : edge 4603 = SearchData.row4603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan180_9 : scan edge (search edge 3) 4 [1093] 1093 innerTail180_9 best = best := by
  rw [innerTail180_9, scan, if_neg (by decide), innerCandidates180_9_checked]
  change scan edge (search edge 3) 4 [1093] 1093 innerTail180_10
    (if 3 ≤ innerCandidates180_9.length then search edge 3 [1093, 4603] innerCandidates180_9 5696 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan180_10

def innerTail180_8 : List Nat := 4447 :: innerTail180_9
def innerCandidates180_8 : List Nat := [6427, 7057]
set_option maxRecDepth 4096 in
theorem innerCandidates180_8_checked : candidates edge 4 1093 4447 innerTail180_9 best.bound = innerCandidates180_8 := by
  unfold candidates
  have row : edge 4447 = SearchData.row4447.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan180_8 : scan edge (search edge 3) 4 [1093] 1093 innerTail180_8 best = best := by
  rw [innerTail180_8, scan, if_neg (by decide), innerCandidates180_8_checked]
  change scan edge (search edge 3) 4 [1093] 1093 innerTail180_9
    (if 3 ≤ innerCandidates180_8.length then search edge 3 [1093, 4447] innerCandidates180_8 5540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan180_9

def innerTail180_7 : List Nat := 4243 :: innerTail180_8
def innerCandidates180_7 : List Nat := [6553]
set_option maxRecDepth 4096 in
theorem innerCandidates180_7_checked : candidates edge 4 1093 4243 innerTail180_8 best.bound = innerCandidates180_7 := by
  unfold candidates
  have row : edge 4243 = SearchData.row4243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan180_7 : scan edge (search edge 3) 4 [1093] 1093 innerTail180_7 best = best := by
  rw [innerTail180_7, scan, if_neg (by decide), innerCandidates180_7_checked]
  change scan edge (search edge 3) 4 [1093] 1093 innerTail180_8
    (if 3 ≤ innerCandidates180_7.length then search edge 3 [1093, 4243] innerCandidates180_7 5336 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan180_8

def innerTail180_6 : List Nat := 4201 :: innerTail180_7
def innerCandidates180_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates180_6_checked : candidates edge 4 1093 4201 innerTail180_7 best.bound = innerCandidates180_6 := by
  unfold candidates
  have row : edge 4201 = SearchData.row4201.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan180_6 : scan edge (search edge 3) 4 [1093] 1093 innerTail180_6 best = best := by
  rw [innerTail180_6, scan, if_neg (by decide), innerCandidates180_6_checked]
  change scan edge (search edge 3) 4 [1093] 1093 innerTail180_7
    (if 3 ≤ innerCandidates180_6.length then search edge 3 [1093, 4201] innerCandidates180_6 5294 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan180_7

def innerTail180_5 : List Nat := 3331 :: innerTail180_6
def innerCandidates180_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates180_5_checked : candidates edge 4 1093 3331 innerTail180_6 best.bound = innerCandidates180_5 := by
  unfold candidates
  have row : edge 3331 = SearchData.row3331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan180_5 : scan edge (search edge 3) 4 [1093] 1093 innerTail180_5 best = best := by
  rw [innerTail180_5, scan, if_neg (by decide), innerCandidates180_5_checked]
  change scan edge (search edge 3) 4 [1093] 1093 innerTail180_6
    (if 3 ≤ innerCandidates180_5.length then search edge 3 [1093, 3331] innerCandidates180_5 4424 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan180_6

def innerTail180_4 : List Nat := 3061 :: innerTail180_5
def innerCandidates180_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates180_4_checked : candidates edge 4 1093 3061 innerTail180_5 best.bound = innerCandidates180_4 := by
  unfold candidates
  have row : edge 3061 = SearchData.row3061.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan180_4 : scan edge (search edge 3) 4 [1093] 1093 innerTail180_4 best = best := by
  rw [innerTail180_4, scan, if_neg (by decide), innerCandidates180_4_checked]
  change scan edge (search edge 3) 4 [1093] 1093 innerTail180_5
    (if 3 ≤ innerCandidates180_4.length then search edge 3 [1093, 3061] innerCandidates180_4 4154 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan180_5

def innerTail180_3 : List Nat := 2437 :: innerTail180_4
def innerCandidates180_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates180_3_checked : candidates edge 4 1093 2437 innerTail180_4 best.bound = innerCandidates180_3 := by
  unfold candidates
  have row : edge 2437 = PrimePairCache.smallRow2437.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan180_3 : scan edge (search edge 3) 4 [1093] 1093 innerTail180_3 best = best := by
  rw [innerTail180_3, scan, if_neg (by decide), innerCandidates180_3_checked]
  change scan edge (search edge 3) 4 [1093] 1093 innerTail180_4
    (if 3 ≤ innerCandidates180_3.length then search edge 3 [1093, 2437] innerCandidates180_3 3530 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan180_4

def innerTail180_2 : List Nat := 1861 :: innerTail180_3
def innerCandidates180_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates180_2_checked : candidates edge 4 1093 1861 innerTail180_3 best.bound = innerCandidates180_2 := by
  unfold candidates
  have row : edge 1861 = PrimePairCache.smallRow1861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan180_2 : scan edge (search edge 3) 4 [1093] 1093 innerTail180_2 best = best := by
  rw [innerTail180_2, scan, if_neg (by decide), innerCandidates180_2_checked]
  change scan edge (search edge 3) 4 [1093] 1093 innerTail180_3
    (if 3 ≤ innerCandidates180_2.length then search edge 3 [1093, 1861] innerCandidates180_2 2954 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan180_3

def innerTail180_1 : List Nat := 1549 :: innerTail180_2
def innerCandidates180_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates180_1_checked : candidates edge 4 1093 1549 innerTail180_2 best.bound = innerCandidates180_1 := by
  unfold candidates
  have row : edge 1549 = PrimePairCache.smallRow1549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan180_1 : scan edge (search edge 3) 4 [1093] 1093 innerTail180_1 best = best := by
  rw [innerTail180_1, scan, if_neg (by decide), innerCandidates180_1_checked]
  change scan edge (search edge 3) 4 [1093] 1093 innerTail180_2
    (if 3 ≤ innerCandidates180_1.length then search edge 3 [1093, 1549] innerCandidates180_1 2642 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan180_2

def innerTail180_0 : List Nat := 1471 :: innerTail180_1
def innerCandidates180_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates180_0_checked : candidates edge 4 1093 1471 innerTail180_1 best.bound = innerCandidates180_0 := by
  unfold candidates
  have row : edge 1471 = PrimePairCache.smallRow1471.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan180_0 : scan edge (search edge 3) 4 [1093] 1093 innerTail180_0 best = best := by
  rw [innerTail180_0, scan, if_neg (by decide), innerCandidates180_0_checked]
  change scan edge (search edge 3) 4 [1093] 1093 innerTail180_1
    (if 3 ≤ innerCandidates180_0.length then search edge 3 [1093, 1471] innerCandidates180_0 2564 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan180_1

theorem rootBranch180 : search edge 4 [1093] rootCandidates180 1093 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1093] 1093 innerTail180_0 best = best
  exact innerScan180_0

end ProjectEulerSolutions.PrimePairUpper.Search
