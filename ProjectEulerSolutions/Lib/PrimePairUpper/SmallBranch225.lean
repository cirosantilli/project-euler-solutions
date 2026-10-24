import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates52 : List Nat := [263, 293, 1277, 1451, 1619, 1697, 1721, 1823, 1949, 2027, 2243, 2267, 2393, 2621, 2741, 2939, 3023, 3209, 3413, 3833, 4337, 4409, 4889, 4967, 5021, 5351, 5939, 6983, 7079, 7283, 7907, 8273]
set_option maxRecDepth 4096 in
theorem rootCandidates52_checked : candidates edge 5 0 257 tail53 best.bound = rootCandidates52 := by
  unfold candidates
  have row : edge 257 = PrimePairCache.smallRow257.contains := by funext q; rfl
  rw [row]
  decide

def innerTail52_32 : List Nat := []
theorem innerScan52_32 : scan edge (search edge 3) 4 [257] 257 innerTail52_32 best = best := rfl

def innerTail52_31 : List Nat := 8273 :: innerTail52_32
def innerCandidates52_31 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates52_31_checked : candidates edge 4 257 8273 innerTail52_32 best.bound = innerCandidates52_31 := by
  unfold candidates
  have row : edge 8273 = SearchData.row8273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan52_31 : scan edge (search edge 3) 4 [257] 257 innerTail52_31 best = best := by
  rw [innerTail52_31, scan, if_neg (by decide), innerCandidates52_31_checked]
  change scan edge (search edge 3) 4 [257] 257 innerTail52_32
    (if 3 ≤ innerCandidates52_31.length then search edge 3 [257, 8273] innerCandidates52_31 8530 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan52_32

def innerTail52_30 : List Nat := 7907 :: innerTail52_31
def innerCandidates52_30 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates52_30_checked : candidates edge 4 257 7907 innerTail52_31 best.bound = innerCandidates52_30 := by
  unfold candidates
  have row : edge 7907 = SearchData.row7907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan52_30 : scan edge (search edge 3) 4 [257] 257 innerTail52_30 best = best := by
  rw [innerTail52_30, scan, if_neg (by decide), innerCandidates52_30_checked]
  change scan edge (search edge 3) 4 [257] 257 innerTail52_31
    (if 3 ≤ innerCandidates52_30.length then search edge 3 [257, 7907] innerCandidates52_30 8164 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan52_31

def innerTail52_29 : List Nat := 7283 :: innerTail52_30
def innerCandidates52_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates52_29_checked : candidates edge 4 257 7283 innerTail52_30 best.bound = innerCandidates52_29 := by
  unfold candidates
  have row : edge 7283 = SearchData.row7283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan52_29 : scan edge (search edge 3) 4 [257] 257 innerTail52_29 best = best := by
  rw [innerTail52_29, scan, if_neg (by decide), innerCandidates52_29_checked]
  change scan edge (search edge 3) 4 [257] 257 innerTail52_30
    (if 3 ≤ innerCandidates52_29.length then search edge 3 [257, 7283] innerCandidates52_29 7540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan52_30

def innerTail52_28 : List Nat := 7079 :: innerTail52_29
def innerCandidates52_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates52_28_checked : candidates edge 4 257 7079 innerTail52_29 best.bound = innerCandidates52_28 := by
  unfold candidates
  have row : edge 7079 = SearchData.row7079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan52_28 : scan edge (search edge 3) 4 [257] 257 innerTail52_28 best = best := by
  rw [innerTail52_28, scan, if_neg (by decide), innerCandidates52_28_checked]
  change scan edge (search edge 3) 4 [257] 257 innerTail52_29
    (if 3 ≤ innerCandidates52_28.length then search edge 3 [257, 7079] innerCandidates52_28 7336 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan52_29

def innerTail52_27 : List Nat := 6983 :: innerTail52_28
def innerCandidates52_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates52_27_checked : candidates edge 4 257 6983 innerTail52_28 best.bound = innerCandidates52_27 := by
  unfold candidates
  have row : edge 6983 = SearchData.row6983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan52_27 : scan edge (search edge 3) 4 [257] 257 innerTail52_27 best = best := by
  rw [innerTail52_27, scan, if_neg (by decide), innerCandidates52_27_checked]
  change scan edge (search edge 3) 4 [257] 257 innerTail52_28
    (if 3 ≤ innerCandidates52_27.length then search edge 3 [257, 6983] innerCandidates52_27 7240 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan52_28

def innerTail52_26 : List Nat := 5939 :: innerTail52_27
def innerCandidates52_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates52_26_checked : candidates edge 4 257 5939 innerTail52_27 best.bound = innerCandidates52_26 := by
  unfold candidates
  have row : edge 5939 = SearchData.row5939.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan52_26 : scan edge (search edge 3) 4 [257] 257 innerTail52_26 best = best := by
  rw [innerTail52_26, scan, if_neg (by decide), innerCandidates52_26_checked]
  change scan edge (search edge 3) 4 [257] 257 innerTail52_27
    (if 3 ≤ innerCandidates52_26.length then search edge 3 [257, 5939] innerCandidates52_26 6196 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan52_27

def innerTail52_25 : List Nat := 5351 :: innerTail52_26
def innerCandidates52_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates52_25_checked : candidates edge 4 257 5351 innerTail52_26 best.bound = innerCandidates52_25 := by
  unfold candidates
  have row : edge 5351 = SearchData.row5351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan52_25 : scan edge (search edge 3) 4 [257] 257 innerTail52_25 best = best := by
  rw [innerTail52_25, scan, if_neg (by decide), innerCandidates52_25_checked]
  change scan edge (search edge 3) 4 [257] 257 innerTail52_26
    (if 3 ≤ innerCandidates52_25.length then search edge 3 [257, 5351] innerCandidates52_25 5608 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan52_26

def innerTail52_24 : List Nat := 5021 :: innerTail52_25
def innerCandidates52_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates52_24_checked : candidates edge 4 257 5021 innerTail52_25 best.bound = innerCandidates52_24 := by
  unfold candidates
  have row : edge 5021 = SearchData.row5021.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan52_24 : scan edge (search edge 3) 4 [257] 257 innerTail52_24 best = best := by
  rw [innerTail52_24, scan, if_neg (by decide), innerCandidates52_24_checked]
  change scan edge (search edge 3) 4 [257] 257 innerTail52_25
    (if 3 ≤ innerCandidates52_24.length then search edge 3 [257, 5021] innerCandidates52_24 5278 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan52_25

def innerTail52_23 : List Nat := 4967 :: innerTail52_24
def innerCandidates52_23 : List Nat := [5939]
set_option maxRecDepth 4096 in
theorem innerCandidates52_23_checked : candidates edge 4 257 4967 innerTail52_24 best.bound = innerCandidates52_23 := by
  unfold candidates
  have row : edge 4967 = SearchData.row4967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan52_23 : scan edge (search edge 3) 4 [257] 257 innerTail52_23 best = best := by
  rw [innerTail52_23, scan, if_neg (by decide), innerCandidates52_23_checked]
  change scan edge (search edge 3) 4 [257] 257 innerTail52_24
    (if 3 ≤ innerCandidates52_23.length then search edge 3 [257, 4967] innerCandidates52_23 5224 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan52_24

def innerTail52_22 : List Nat := 4889 :: innerTail52_23
def innerCandidates52_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates52_22_checked : candidates edge 4 257 4889 innerTail52_23 best.bound = innerCandidates52_22 := by
  unfold candidates
  have row : edge 4889 = SearchData.row4889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan52_22 : scan edge (search edge 3) 4 [257] 257 innerTail52_22 best = best := by
  rw [innerTail52_22, scan, if_neg (by decide), innerCandidates52_22_checked]
  change scan edge (search edge 3) 4 [257] 257 innerTail52_23
    (if 3 ≤ innerCandidates52_22.length then search edge 3 [257, 4889] innerCandidates52_22 5146 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan52_23

def innerTail52_21 : List Nat := 4409 :: innerTail52_22
def innerCandidates52_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates52_21_checked : candidates edge 4 257 4409 innerTail52_22 best.bound = innerCandidates52_21 := by
  unfold candidates
  have row : edge 4409 = SearchData.row4409.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan52_21 : scan edge (search edge 3) 4 [257] 257 innerTail52_21 best = best := by
  rw [innerTail52_21, scan, if_neg (by decide), innerCandidates52_21_checked]
  change scan edge (search edge 3) 4 [257] 257 innerTail52_22
    (if 3 ≤ innerCandidates52_21.length then search edge 3 [257, 4409] innerCandidates52_21 4666 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan52_22

def innerTail52_20 : List Nat := 4337 :: innerTail52_21
def innerCandidates52_20 : List Nat := [4889]
set_option maxRecDepth 4096 in
theorem innerCandidates52_20_checked : candidates edge 4 257 4337 innerTail52_21 best.bound = innerCandidates52_20 := by
  unfold candidates
  have row : edge 4337 = SearchData.row4337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan52_20 : scan edge (search edge 3) 4 [257] 257 innerTail52_20 best = best := by
  rw [innerTail52_20, scan, if_neg (by decide), innerCandidates52_20_checked]
  change scan edge (search edge 3) 4 [257] 257 innerTail52_21
    (if 3 ≤ innerCandidates52_20.length then search edge 3 [257, 4337] innerCandidates52_20 4594 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan52_21

def innerTail52_19 : List Nat := 3833 :: innerTail52_20
def innerCandidates52_19 : List Nat := [4889, 6983]
set_option maxRecDepth 4096 in
theorem innerCandidates52_19_checked : candidates edge 4 257 3833 innerTail52_20 best.bound = innerCandidates52_19 := by
  unfold candidates
  have row : edge 3833 = SearchData.row3833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan52_19 : scan edge (search edge 3) 4 [257] 257 innerTail52_19 best = best := by
  rw [innerTail52_19, scan, if_neg (by decide), innerCandidates52_19_checked]
  change scan edge (search edge 3) 4 [257] 257 innerTail52_20
    (if 3 ≤ innerCandidates52_19.length then search edge 3 [257, 3833] innerCandidates52_19 4090 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan52_20

def innerTail52_18 : List Nat := 3413 :: innerTail52_19
def innerCandidates52_18 : List Nat := [5939]
set_option maxRecDepth 4096 in
theorem innerCandidates52_18_checked : candidates edge 4 257 3413 innerTail52_19 best.bound = innerCandidates52_18 := by
  unfold candidates
  have row : edge 3413 = SearchData.row3413.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan52_18 : scan edge (search edge 3) 4 [257] 257 innerTail52_18 best = best := by
  rw [innerTail52_18, scan, if_neg (by decide), innerCandidates52_18_checked]
  change scan edge (search edge 3) 4 [257] 257 innerTail52_19
    (if 3 ≤ innerCandidates52_18.length then search edge 3 [257, 3413] innerCandidates52_18 3670 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan52_19

def innerTail52_17 : List Nat := 3209 :: innerTail52_18
def innerCandidates52_17 : List Nat := [3833]
set_option maxRecDepth 4096 in
theorem innerCandidates52_17_checked : candidates edge 4 257 3209 innerTail52_18 best.bound = innerCandidates52_17 := by
  unfold candidates
  have row : edge 3209 = SearchData.row3209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan52_17 : scan edge (search edge 3) 4 [257] 257 innerTail52_17 best = best := by
  rw [innerTail52_17, scan, if_neg (by decide), innerCandidates52_17_checked]
  change scan edge (search edge 3) 4 [257] 257 innerTail52_18
    (if 3 ≤ innerCandidates52_17.length then search edge 3 [257, 3209] innerCandidates52_17 3466 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan52_18

def innerTail52_16 : List Nat := 3023 :: innerTail52_17
def innerCandidates52_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates52_16_checked : candidates edge 4 257 3023 innerTail52_17 best.bound = innerCandidates52_16 := by
  unfold candidates
  have row : edge 3023 = SearchData.row3023.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan52_16 : scan edge (search edge 3) 4 [257] 257 innerTail52_16 best = best := by
  rw [innerTail52_16, scan, if_neg (by decide), innerCandidates52_16_checked]
  change scan edge (search edge 3) 4 [257] 257 innerTail52_17
    (if 3 ≤ innerCandidates52_16.length then search edge 3 [257, 3023] innerCandidates52_16 3280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan52_17

def innerTail52_15 : List Nat := 2939 :: innerTail52_16
def innerCandidates52_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates52_15_checked : candidates edge 4 257 2939 innerTail52_16 best.bound = innerCandidates52_15 := by
  unfold candidates
  have row : edge 2939 = SearchData.row2939.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan52_15 : scan edge (search edge 3) 4 [257] 257 innerTail52_15 best = best := by
  rw [innerTail52_15, scan, if_neg (by decide), innerCandidates52_15_checked]
  change scan edge (search edge 3) 4 [257] 257 innerTail52_16
    (if 3 ≤ innerCandidates52_15.length then search edge 3 [257, 2939] innerCandidates52_15 3196 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan52_16

def innerTail52_14 : List Nat := 2741 :: innerTail52_15
def innerCandidates52_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates52_14_checked : candidates edge 4 257 2741 innerTail52_15 best.bound = innerCandidates52_14 := by
  unfold candidates
  have row : edge 2741 = SearchData.row2741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan52_14 : scan edge (search edge 3) 4 [257] 257 innerTail52_14 best = best := by
  rw [innerTail52_14, scan, if_neg (by decide), innerCandidates52_14_checked]
  change scan edge (search edge 3) 4 [257] 257 innerTail52_15
    (if 3 ≤ innerCandidates52_14.length then search edge 3 [257, 2741] innerCandidates52_14 2998 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan52_15

def innerTail52_13 : List Nat := 2621 :: innerTail52_14
def innerCandidates52_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates52_13_checked : candidates edge 4 257 2621 innerTail52_14 best.bound = innerCandidates52_13 := by
  unfold candidates
  have row : edge 2621 = PrimePairCache.smallRow2621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan52_13 : scan edge (search edge 3) 4 [257] 257 innerTail52_13 best = best := by
  rw [innerTail52_13, scan, if_neg (by decide), innerCandidates52_13_checked]
  change scan edge (search edge 3) 4 [257] 257 innerTail52_14
    (if 3 ≤ innerCandidates52_13.length then search edge 3 [257, 2621] innerCandidates52_13 2878 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan52_14

def innerTail52_12 : List Nat := 2393 :: innerTail52_13
def innerCandidates52_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates52_12_checked : candidates edge 4 257 2393 innerTail52_13 best.bound = innerCandidates52_12 := by
  unfold candidates
  have row : edge 2393 = PrimePairCache.smallRow2393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan52_12 : scan edge (search edge 3) 4 [257] 257 innerTail52_12 best = best := by
  rw [innerTail52_12, scan, if_neg (by decide), innerCandidates52_12_checked]
  change scan edge (search edge 3) 4 [257] 257 innerTail52_13
    (if 3 ≤ innerCandidates52_12.length then search edge 3 [257, 2393] innerCandidates52_12 2650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan52_13

def innerTail52_11 : List Nat := 2267 :: innerTail52_12
def innerCandidates52_11 : List Nat := [2939]
set_option maxRecDepth 4096 in
theorem innerCandidates52_11_checked : candidates edge 4 257 2267 innerTail52_12 best.bound = innerCandidates52_11 := by
  unfold candidates
  have row : edge 2267 = PrimePairCache.smallRow2267.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan52_11 : scan edge (search edge 3) 4 [257] 257 innerTail52_11 best = best := by
  rw [innerTail52_11, scan, if_neg (by decide), innerCandidates52_11_checked]
  change scan edge (search edge 3) 4 [257] 257 innerTail52_12
    (if 3 ≤ innerCandidates52_11.length then search edge 3 [257, 2267] innerCandidates52_11 2524 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan52_12

def innerTail52_10 : List Nat := 2243 :: innerTail52_11
def innerCandidates52_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates52_10_checked : candidates edge 4 257 2243 innerTail52_11 best.bound = innerCandidates52_10 := by
  unfold candidates
  have row : edge 2243 = PrimePairCache.smallRow2243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan52_10 : scan edge (search edge 3) 4 [257] 257 innerTail52_10 best = best := by
  rw [innerTail52_10, scan, if_neg (by decide), innerCandidates52_10_checked]
  change scan edge (search edge 3) 4 [257] 257 innerTail52_11
    (if 3 ≤ innerCandidates52_10.length then search edge 3 [257, 2243] innerCandidates52_10 2500 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan52_11

def innerTail52_9 : List Nat := 2027 :: innerTail52_10
def innerCandidates52_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates52_9_checked : candidates edge 4 257 2027 innerTail52_10 best.bound = innerCandidates52_9 := by
  unfold candidates
  have row : edge 2027 = PrimePairCache.smallRow2027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan52_9 : scan edge (search edge 3) 4 [257] 257 innerTail52_9 best = best := by
  rw [innerTail52_9, scan, if_neg (by decide), innerCandidates52_9_checked]
  change scan edge (search edge 3) 4 [257] 257 innerTail52_10
    (if 3 ≤ innerCandidates52_9.length then search edge 3 [257, 2027] innerCandidates52_9 2284 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan52_10

def innerTail52_8 : List Nat := 1949 :: innerTail52_9
def innerCandidates52_8 : List Nat := [2027, 2621, 4337]
set_option maxRecDepth 4096 in
theorem innerCandidates52_8_checked : candidates edge 4 257 1949 innerTail52_9 best.bound = innerCandidates52_8 := by
  unfold candidates
  have row : edge 1949 = PrimePairCache.smallRow1949.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild52_8 : search edge 3 [257, 1949] innerCandidates52_8 2206 best = best := by decide

theorem innerScan52_8 : scan edge (search edge 3) 4 [257] 257 innerTail52_8 best = best := by
  rw [innerTail52_8, scan, if_neg (by decide), innerCandidates52_8_checked]
  change scan edge (search edge 3) 4 [257] 257 innerTail52_9
    (if 3 ≤ innerCandidates52_8.length then search edge 3 [257, 1949] innerCandidates52_8 2206 best else best) = best
  rw [if_pos (by decide), innerChild52_8]
  exact innerScan52_9

def innerTail52_7 : List Nat := 1823 :: innerTail52_8
def innerCandidates52_7 : List Nat := [3023, 6983]
set_option maxRecDepth 4096 in
theorem innerCandidates52_7_checked : candidates edge 4 257 1823 innerTail52_8 best.bound = innerCandidates52_7 := by
  unfold candidates
  have row : edge 1823 = PrimePairCache.smallRow1823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan52_7 : scan edge (search edge 3) 4 [257] 257 innerTail52_7 best = best := by
  rw [innerTail52_7, scan, if_neg (by decide), innerCandidates52_7_checked]
  change scan edge (search edge 3) 4 [257] 257 innerTail52_8
    (if 3 ≤ innerCandidates52_7.length then search edge 3 [257, 1823] innerCandidates52_7 2080 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan52_8

def innerTail52_6 : List Nat := 1721 :: innerTail52_7
def innerCandidates52_6 : List Nat := [2621, 7283, 8273]
set_option maxRecDepth 4096 in
theorem innerCandidates52_6_checked : candidates edge 4 257 1721 innerTail52_7 best.bound = innerCandidates52_6 := by
  unfold candidates
  have row : edge 1721 = PrimePairCache.smallRow1721.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild52_6 : search edge 3 [257, 1721] innerCandidates52_6 1978 best = best := by decide

theorem innerScan52_6 : scan edge (search edge 3) 4 [257] 257 innerTail52_6 best = best := by
  rw [innerTail52_6, scan, if_neg (by decide), innerCandidates52_6_checked]
  change scan edge (search edge 3) 4 [257] 257 innerTail52_7
    (if 3 ≤ innerCandidates52_6.length then search edge 3 [257, 1721] innerCandidates52_6 1978 best else best) = best
  rw [if_pos (by decide), innerChild52_6]
  exact innerScan52_7

def innerTail52_5 : List Nat := 1697 :: innerTail52_6
def innerCandidates52_5 : List Nat := [2027, 2741]
set_option maxRecDepth 4096 in
theorem innerCandidates52_5_checked : candidates edge 4 257 1697 innerTail52_6 best.bound = innerCandidates52_5 := by
  unfold candidates
  have row : edge 1697 = PrimePairCache.smallRow1697.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan52_5 : scan edge (search edge 3) 4 [257] 257 innerTail52_5 best = best := by
  rw [innerTail52_5, scan, if_neg (by decide), innerCandidates52_5_checked]
  change scan edge (search edge 3) 4 [257] 257 innerTail52_6
    (if 3 ≤ innerCandidates52_5.length then search edge 3 [257, 1697] innerCandidates52_5 1954 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan52_6

def innerTail52_4 : List Nat := 1619 :: innerTail52_5
def innerCandidates52_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates52_4_checked : candidates edge 4 257 1619 innerTail52_5 best.bound = innerCandidates52_4 := by
  unfold candidates
  have row : edge 1619 = PrimePairCache.smallRow1619.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan52_4 : scan edge (search edge 3) 4 [257] 257 innerTail52_4 best = best := by
  rw [innerTail52_4, scan, if_neg (by decide), innerCandidates52_4_checked]
  change scan edge (search edge 3) 4 [257] 257 innerTail52_5
    (if 3 ≤ innerCandidates52_4.length then search edge 3 [257, 1619] innerCandidates52_4 1876 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan52_5

def innerTail52_3 : List Nat := 1451 :: innerTail52_4
def innerCandidates52_3 : List Nat := [3413, 7907]
set_option maxRecDepth 4096 in
theorem innerCandidates52_3_checked : candidates edge 4 257 1451 innerTail52_4 best.bound = innerCandidates52_3 := by
  unfold candidates
  have row : edge 1451 = PrimePairCache.smallRow1451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan52_3 : scan edge (search edge 3) 4 [257] 257 innerTail52_3 best = best := by
  rw [innerTail52_3, scan, if_neg (by decide), innerCandidates52_3_checked]
  change scan edge (search edge 3) 4 [257] 257 innerTail52_4
    (if 3 ≤ innerCandidates52_3.length then search edge 3 [257, 1451] innerCandidates52_3 1708 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan52_4

def innerTail52_2 : List Nat := 1277 :: innerTail52_3
def innerCandidates52_2 : List Nat := [4889, 4967]
set_option maxRecDepth 4096 in
theorem innerCandidates52_2_checked : candidates edge 4 257 1277 innerTail52_3 best.bound = innerCandidates52_2 := by
  unfold candidates
  have row : edge 1277 = PrimePairCache.smallRow1277.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan52_2 : scan edge (search edge 3) 4 [257] 257 innerTail52_2 best = best := by
  rw [innerTail52_2, scan, if_neg (by decide), innerCandidates52_2_checked]
  change scan edge (search edge 3) 4 [257] 257 innerTail52_3
    (if 3 ≤ innerCandidates52_2.length then search edge 3 [257, 1277] innerCandidates52_2 1534 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan52_3

def innerTail52_1 : List Nat := 293 :: innerTail52_2
def innerCandidates52_1 : List Nat := [1823]
set_option maxRecDepth 4096 in
theorem innerCandidates52_1_checked : candidates edge 4 257 293 innerTail52_2 best.bound = innerCandidates52_1 := by
  unfold candidates
  have row : edge 293 = PrimePairCache.smallRow293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan52_1 : scan edge (search edge 3) 4 [257] 257 innerTail52_1 best = best := by
  rw [innerTail52_1, scan, if_neg (by decide), innerCandidates52_1_checked]
  change scan edge (search edge 3) 4 [257] 257 innerTail52_2
    (if 3 ≤ innerCandidates52_1.length then search edge 3 [257, 293] innerCandidates52_1 550 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan52_2

def innerTail52_0 : List Nat := 263 :: innerTail52_1
def innerCandidates52_0 : List Nat := [293]
set_option maxRecDepth 4096 in
theorem innerCandidates52_0_checked : candidates edge 4 257 263 innerTail52_1 best.bound = innerCandidates52_0 := by
  unfold candidates
  have row : edge 263 = PrimePairCache.smallRow263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan52_0 : scan edge (search edge 3) 4 [257] 257 innerTail52_0 best = best := by
  rw [innerTail52_0, scan, if_neg (by decide), innerCandidates52_0_checked]
  change scan edge (search edge 3) 4 [257] 257 innerTail52_1
    (if 3 ≤ innerCandidates52_0.length then search edge 3 [257, 263] innerCandidates52_0 520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan52_1

theorem rootBranch52 : search edge 4 [257] rootCandidates52 257 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [257] 257 innerTail52_0 best = best
  exact innerScan52_0

end ProjectEulerSolutions.PrimePairUpper.Search
