import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates60 : List Nat := [367, 523, 577, 631, 733, 1069, 1093, 1279, 1381, 1993, 2137, 2269, 2467, 3547, 3943, 4243, 5119, 5209, 5857, 6373, 6679, 7159, 7681, 7927, 8059]
set_option maxRecDepth 4096 in
theorem rootCandidates60_checked : candidates edge 5 0 307 tail61 best.bound = rootCandidates60 := by
  unfold candidates
  have row : edge 307 = PrimePairCache.smallRow307.contains := by funext q; rfl
  rw [row]
  decide

def innerTail60_25 : List Nat := []
theorem innerScan60_25 : scan edge (search edge 3) 4 [307] 307 innerTail60_25 best = best := rfl

def innerTail60_24 : List Nat := 8059 :: innerTail60_25
def innerCandidates60_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates60_24_checked : candidates edge 4 307 8059 innerTail60_25 best.bound = innerCandidates60_24 := by
  unfold candidates
  have row : edge 8059 = SearchData.row8059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan60_24 : scan edge (search edge 3) 4 [307] 307 innerTail60_24 best = best := by
  rw [innerTail60_24, scan, if_neg (by decide), innerCandidates60_24_checked]
  change scan edge (search edge 3) 4 [307] 307 innerTail60_25
    (if 3 ≤ innerCandidates60_24.length then search edge 3 [307, 8059] innerCandidates60_24 8366 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan60_25

def innerTail60_23 : List Nat := 7927 :: innerTail60_24
def innerCandidates60_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates60_23_checked : candidates edge 4 307 7927 innerTail60_24 best.bound = innerCandidates60_23 := by
  unfold candidates
  have row : edge 7927 = SearchData.row7927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan60_23 : scan edge (search edge 3) 4 [307] 307 innerTail60_23 best = best := by
  rw [innerTail60_23, scan, if_neg (by decide), innerCandidates60_23_checked]
  change scan edge (search edge 3) 4 [307] 307 innerTail60_24
    (if 3 ≤ innerCandidates60_23.length then search edge 3 [307, 7927] innerCandidates60_23 8234 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan60_24

def innerTail60_22 : List Nat := 7681 :: innerTail60_23
def innerCandidates60_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates60_22_checked : candidates edge 4 307 7681 innerTail60_23 best.bound = innerCandidates60_22 := by
  unfold candidates
  have row : edge 7681 = SearchData.row7681.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan60_22 : scan edge (search edge 3) 4 [307] 307 innerTail60_22 best = best := by
  rw [innerTail60_22, scan, if_neg (by decide), innerCandidates60_22_checked]
  change scan edge (search edge 3) 4 [307] 307 innerTail60_23
    (if 3 ≤ innerCandidates60_22.length then search edge 3 [307, 7681] innerCandidates60_22 7988 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan60_23

def innerTail60_21 : List Nat := 7159 :: innerTail60_22
def innerCandidates60_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates60_21_checked : candidates edge 4 307 7159 innerTail60_22 best.bound = innerCandidates60_21 := by
  unfold candidates
  have row : edge 7159 = SearchData.row7159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan60_21 : scan edge (search edge 3) 4 [307] 307 innerTail60_21 best = best := by
  rw [innerTail60_21, scan, if_neg (by decide), innerCandidates60_21_checked]
  change scan edge (search edge 3) 4 [307] 307 innerTail60_22
    (if 3 ≤ innerCandidates60_21.length then search edge 3 [307, 7159] innerCandidates60_21 7466 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan60_22

def innerTail60_20 : List Nat := 6679 :: innerTail60_21
def innerCandidates60_20 : List Nat := [8059]
set_option maxRecDepth 4096 in
theorem innerCandidates60_20_checked : candidates edge 4 307 6679 innerTail60_21 best.bound = innerCandidates60_20 := by
  unfold candidates
  have row : edge 6679 = SearchData.row6679.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan60_20 : scan edge (search edge 3) 4 [307] 307 innerTail60_20 best = best := by
  rw [innerTail60_20, scan, if_neg (by decide), innerCandidates60_20_checked]
  change scan edge (search edge 3) 4 [307] 307 innerTail60_21
    (if 3 ≤ innerCandidates60_20.length then search edge 3 [307, 6679] innerCandidates60_20 6986 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan60_21

def innerTail60_19 : List Nat := 6373 :: innerTail60_20
def innerCandidates60_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates60_19_checked : candidates edge 4 307 6373 innerTail60_20 best.bound = innerCandidates60_19 := by
  unfold candidates
  have row : edge 6373 = SearchData.row6373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan60_19 : scan edge (search edge 3) 4 [307] 307 innerTail60_19 best = best := by
  rw [innerTail60_19, scan, if_neg (by decide), innerCandidates60_19_checked]
  change scan edge (search edge 3) 4 [307] 307 innerTail60_20
    (if 3 ≤ innerCandidates60_19.length then search edge 3 [307, 6373] innerCandidates60_19 6680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan60_20

def innerTail60_18 : List Nat := 5857 :: innerTail60_19
def innerCandidates60_18 : List Nat := [8059]
set_option maxRecDepth 4096 in
theorem innerCandidates60_18_checked : candidates edge 4 307 5857 innerTail60_19 best.bound = innerCandidates60_18 := by
  unfold candidates
  have row : edge 5857 = SearchData.row5857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan60_18 : scan edge (search edge 3) 4 [307] 307 innerTail60_18 best = best := by
  rw [innerTail60_18, scan, if_neg (by decide), innerCandidates60_18_checked]
  change scan edge (search edge 3) 4 [307] 307 innerTail60_19
    (if 3 ≤ innerCandidates60_18.length then search edge 3 [307, 5857] innerCandidates60_18 6164 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan60_19

def innerTail60_17 : List Nat := 5209 :: innerTail60_18
def innerCandidates60_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates60_17_checked : candidates edge 4 307 5209 innerTail60_18 best.bound = innerCandidates60_17 := by
  unfold candidates
  have row : edge 5209 = SearchData.row5209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan60_17 : scan edge (search edge 3) 4 [307] 307 innerTail60_17 best = best := by
  rw [innerTail60_17, scan, if_neg (by decide), innerCandidates60_17_checked]
  change scan edge (search edge 3) 4 [307] 307 innerTail60_18
    (if 3 ≤ innerCandidates60_17.length then search edge 3 [307, 5209] innerCandidates60_17 5516 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan60_18

def innerTail60_16 : List Nat := 5119 :: innerTail60_17
def innerCandidates60_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates60_16_checked : candidates edge 4 307 5119 innerTail60_17 best.bound = innerCandidates60_16 := by
  unfold candidates
  have row : edge 5119 = SearchData.row5119.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan60_16 : scan edge (search edge 3) 4 [307] 307 innerTail60_16 best = best := by
  rw [innerTail60_16, scan, if_neg (by decide), innerCandidates60_16_checked]
  change scan edge (search edge 3) 4 [307] 307 innerTail60_17
    (if 3 ≤ innerCandidates60_16.length then search edge 3 [307, 5119] innerCandidates60_16 5426 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan60_17

def innerTail60_15 : List Nat := 4243 :: innerTail60_16
def innerCandidates60_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates60_15_checked : candidates edge 4 307 4243 innerTail60_16 best.bound = innerCandidates60_15 := by
  unfold candidates
  have row : edge 4243 = SearchData.row4243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan60_15 : scan edge (search edge 3) 4 [307] 307 innerTail60_15 best = best := by
  rw [innerTail60_15, scan, if_neg (by decide), innerCandidates60_15_checked]
  change scan edge (search edge 3) 4 [307] 307 innerTail60_16
    (if 3 ≤ innerCandidates60_15.length then search edge 3 [307, 4243] innerCandidates60_15 4550 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan60_16

def innerTail60_14 : List Nat := 3943 :: innerTail60_15
def innerCandidates60_14 : List Nat := [8059]
set_option maxRecDepth 4096 in
theorem innerCandidates60_14_checked : candidates edge 4 307 3943 innerTail60_15 best.bound = innerCandidates60_14 := by
  unfold candidates
  have row : edge 3943 = SearchData.row3943.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan60_14 : scan edge (search edge 3) 4 [307] 307 innerTail60_14 best = best := by
  rw [innerTail60_14, scan, if_neg (by decide), innerCandidates60_14_checked]
  change scan edge (search edge 3) 4 [307] 307 innerTail60_15
    (if 3 ≤ innerCandidates60_14.length then search edge 3 [307, 3943] innerCandidates60_14 4250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan60_15

def innerTail60_13 : List Nat := 3547 :: innerTail60_14
def innerCandidates60_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates60_13_checked : candidates edge 4 307 3547 innerTail60_14 best.bound = innerCandidates60_13 := by
  unfold candidates
  have row : edge 3547 = SearchData.row3547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan60_13 : scan edge (search edge 3) 4 [307] 307 innerTail60_13 best = best := by
  rw [innerTail60_13, scan, if_neg (by decide), innerCandidates60_13_checked]
  change scan edge (search edge 3) 4 [307] 307 innerTail60_14
    (if 3 ≤ innerCandidates60_13.length then search edge 3 [307, 3547] innerCandidates60_13 3854 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan60_14

def innerTail60_12 : List Nat := 2467 :: innerTail60_13
def innerCandidates60_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates60_12_checked : candidates edge 4 307 2467 innerTail60_13 best.bound = innerCandidates60_12 := by
  unfold candidates
  have row : edge 2467 = PrimePairCache.smallRow2467.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan60_12 : scan edge (search edge 3) 4 [307] 307 innerTail60_12 best = best := by
  rw [innerTail60_12, scan, if_neg (by decide), innerCandidates60_12_checked]
  change scan edge (search edge 3) 4 [307] 307 innerTail60_13
    (if 3 ≤ innerCandidates60_12.length then search edge 3 [307, 2467] innerCandidates60_12 2774 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan60_13

def innerTail60_11 : List Nat := 2269 :: innerTail60_12
def innerCandidates60_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates60_11_checked : candidates edge 4 307 2269 innerTail60_12 best.bound = innerCandidates60_11 := by
  unfold candidates
  have row : edge 2269 = PrimePairCache.smallRow2269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan60_11 : scan edge (search edge 3) 4 [307] 307 innerTail60_11 best = best := by
  rw [innerTail60_11, scan, if_neg (by decide), innerCandidates60_11_checked]
  change scan edge (search edge 3) 4 [307] 307 innerTail60_12
    (if 3 ≤ innerCandidates60_11.length then search edge 3 [307, 2269] innerCandidates60_11 2576 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan60_12

def innerTail60_10 : List Nat := 2137 :: innerTail60_11
def innerCandidates60_10 : List Nat := [7159]
set_option maxRecDepth 4096 in
theorem innerCandidates60_10_checked : candidates edge 4 307 2137 innerTail60_11 best.bound = innerCandidates60_10 := by
  unfold candidates
  have row : edge 2137 = PrimePairCache.smallRow2137.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan60_10 : scan edge (search edge 3) 4 [307] 307 innerTail60_10 best = best := by
  rw [innerTail60_10, scan, if_neg (by decide), innerCandidates60_10_checked]
  change scan edge (search edge 3) 4 [307] 307 innerTail60_11
    (if 3 ≤ innerCandidates60_10.length then search edge 3 [307, 2137] innerCandidates60_10 2444 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan60_11

def innerTail60_9 : List Nat := 1993 :: innerTail60_10
def innerCandidates60_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates60_9_checked : candidates edge 4 307 1993 innerTail60_10 best.bound = innerCandidates60_9 := by
  unfold candidates
  have row : edge 1993 = PrimePairCache.smallRow1993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan60_9 : scan edge (search edge 3) 4 [307] 307 innerTail60_9 best = best := by
  rw [innerTail60_9, scan, if_neg (by decide), innerCandidates60_9_checked]
  change scan edge (search edge 3) 4 [307] 307 innerTail60_10
    (if 3 ≤ innerCandidates60_9.length then search edge 3 [307, 1993] innerCandidates60_9 2300 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan60_10

def innerTail60_8 : List Nat := 1381 :: innerTail60_9
def innerCandidates60_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates60_8_checked : candidates edge 4 307 1381 innerTail60_9 best.bound = innerCandidates60_8 := by
  unfold candidates
  have row : edge 1381 = PrimePairCache.smallRow1381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan60_8 : scan edge (search edge 3) 4 [307] 307 innerTail60_8 best = best := by
  rw [innerTail60_8, scan, if_neg (by decide), innerCandidates60_8_checked]
  change scan edge (search edge 3) 4 [307] 307 innerTail60_9
    (if 3 ≤ innerCandidates60_8.length then search edge 3 [307, 1381] innerCandidates60_8 1688 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan60_9

def innerTail60_7 : List Nat := 1279 :: innerTail60_8
def innerCandidates60_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates60_7_checked : candidates edge 4 307 1279 innerTail60_8 best.bound = innerCandidates60_7 := by
  unfold candidates
  have row : edge 1279 = PrimePairCache.smallRow1279.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan60_7 : scan edge (search edge 3) 4 [307] 307 innerTail60_7 best = best := by
  rw [innerTail60_7, scan, if_neg (by decide), innerCandidates60_7_checked]
  change scan edge (search edge 3) 4 [307] 307 innerTail60_8
    (if 3 ≤ innerCandidates60_7.length then search edge 3 [307, 1279] innerCandidates60_7 1586 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan60_8

def innerTail60_6 : List Nat := 1093 :: innerTail60_7
def innerCandidates60_6 : List Nat := [4243]
set_option maxRecDepth 4096 in
theorem innerCandidates60_6_checked : candidates edge 4 307 1093 innerTail60_7 best.bound = innerCandidates60_6 := by
  unfold candidates
  have row : edge 1093 = PrimePairCache.smallRow1093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan60_6 : scan edge (search edge 3) 4 [307] 307 innerTail60_6 best = best := by
  rw [innerTail60_6, scan, if_neg (by decide), innerCandidates60_6_checked]
  change scan edge (search edge 3) 4 [307] 307 innerTail60_7
    (if 3 ≤ innerCandidates60_6.length then search edge 3 [307, 1093] innerCandidates60_6 1400 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan60_7

def innerTail60_5 : List Nat := 1069 :: innerTail60_6
def innerCandidates60_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates60_5_checked : candidates edge 4 307 1069 innerTail60_6 best.bound = innerCandidates60_5 := by
  unfold candidates
  have row : edge 1069 = PrimePairCache.smallRow1069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan60_5 : scan edge (search edge 3) 4 [307] 307 innerTail60_5 best = best := by
  rw [innerTail60_5, scan, if_neg (by decide), innerCandidates60_5_checked]
  change scan edge (search edge 3) 4 [307] 307 innerTail60_6
    (if 3 ≤ innerCandidates60_5.length then search edge 3 [307, 1069] innerCandidates60_5 1376 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan60_6

def innerTail60_4 : List Nat := 733 :: innerTail60_5
def innerCandidates60_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates60_4_checked : candidates edge 4 307 733 innerTail60_5 best.bound = innerCandidates60_4 := by
  unfold candidates
  have row : edge 733 = PrimePairCache.smallRow733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan60_4 : scan edge (search edge 3) 4 [307] 307 innerTail60_4 best = best := by
  rw [innerTail60_4, scan, if_neg (by decide), innerCandidates60_4_checked]
  change scan edge (search edge 3) 4 [307] 307 innerTail60_5
    (if 3 ≤ innerCandidates60_4.length then search edge 3 [307, 733] innerCandidates60_4 1040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan60_5

def innerTail60_3 : List Nat := 631 :: innerTail60_4
def innerCandidates60_3 : List Nat := [7681]
set_option maxRecDepth 4096 in
theorem innerCandidates60_3_checked : candidates edge 4 307 631 innerTail60_4 best.bound = innerCandidates60_3 := by
  unfold candidates
  have row : edge 631 = PrimePairCache.smallRow631.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan60_3 : scan edge (search edge 3) 4 [307] 307 innerTail60_3 best = best := by
  rw [innerTail60_3, scan, if_neg (by decide), innerCandidates60_3_checked]
  change scan edge (search edge 3) 4 [307] 307 innerTail60_4
    (if 3 ≤ innerCandidates60_3.length then search edge 3 [307, 631] innerCandidates60_3 938 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan60_4

def innerTail60_2 : List Nat := 577 :: innerTail60_3
def innerCandidates60_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates60_2_checked : candidates edge 4 307 577 innerTail60_3 best.bound = innerCandidates60_2 := by
  unfold candidates
  have row : edge 577 = PrimePairCache.smallRow577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan60_2 : scan edge (search edge 3) 4 [307] 307 innerTail60_2 best = best := by
  rw [innerTail60_2, scan, if_neg (by decide), innerCandidates60_2_checked]
  change scan edge (search edge 3) 4 [307] 307 innerTail60_3
    (if 3 ≤ innerCandidates60_2.length then search edge 3 [307, 577] innerCandidates60_2 884 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan60_3

def innerTail60_1 : List Nat := 523 :: innerTail60_2
def innerCandidates60_1 : List Nat := [577]
set_option maxRecDepth 4096 in
theorem innerCandidates60_1_checked : candidates edge 4 307 523 innerTail60_2 best.bound = innerCandidates60_1 := by
  unfold candidates
  have row : edge 523 = PrimePairCache.smallRow523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan60_1 : scan edge (search edge 3) 4 [307] 307 innerTail60_1 best = best := by
  rw [innerTail60_1, scan, if_neg (by decide), innerCandidates60_1_checked]
  change scan edge (search edge 3) 4 [307] 307 innerTail60_2
    (if 3 ≤ innerCandidates60_1.length then search edge 3 [307, 523] innerCandidates60_1 830 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan60_2

def innerTail60_0 : List Nat := 367 :: innerTail60_1
def innerCandidates60_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates60_0_checked : candidates edge 4 307 367 innerTail60_1 best.bound = innerCandidates60_0 := by
  unfold candidates
  have row : edge 367 = PrimePairCache.smallRow367.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan60_0 : scan edge (search edge 3) 4 [307] 307 innerTail60_0 best = best := by
  rw [innerTail60_0, scan, if_neg (by decide), innerCandidates60_0_checked]
  change scan edge (search edge 3) 4 [307] 307 innerTail60_1
    (if 3 ≤ innerCandidates60_0.length then search edge 3 [307, 367] innerCandidates60_0 674 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan60_1

theorem rootBranch60 : search edge 4 [307] rootCandidates60 307 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [307] 307 innerTail60_0 best = best
  exact innerScan60_0

def rootCandidates59 : List Nat := [311, 467, 617, 827, 1319, 1433, 1493, 1583, 1823, 1997, 2411, 2459, 2777, 3083, 4091, 4751, 5147, 5171, 5189, 5717, 7523, 7757, 8009, 8237]
set_option maxRecDepth 4096 in
theorem rootCandidates59_checked : candidates edge 5 0 293 tail60 best.bound = rootCandidates59 := by
  unfold candidates
  have row : edge 293 = PrimePairCache.smallRow293.contains := by funext q; rfl
  rw [row]
  decide

def innerTail59_24 : List Nat := []
theorem innerScan59_24 : scan edge (search edge 3) 4 [293] 293 innerTail59_24 best = best := rfl

def innerTail59_23 : List Nat := 8237 :: innerTail59_24
def innerCandidates59_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates59_23_checked : candidates edge 4 293 8237 innerTail59_24 best.bound = innerCandidates59_23 := by
  unfold candidates
  have row : edge 8237 = SearchData.row8237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan59_23 : scan edge (search edge 3) 4 [293] 293 innerTail59_23 best = best := by
  rw [innerTail59_23, scan, if_neg (by decide), innerCandidates59_23_checked]
  change scan edge (search edge 3) 4 [293] 293 innerTail59_24
    (if 3 ≤ innerCandidates59_23.length then search edge 3 [293, 8237] innerCandidates59_23 8530 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan59_24

def innerTail59_22 : List Nat := 8009 :: innerTail59_23
def innerCandidates59_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates59_22_checked : candidates edge 4 293 8009 innerTail59_23 best.bound = innerCandidates59_22 := by
  unfold candidates
  have row : edge 8009 = SearchData.row8009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan59_22 : scan edge (search edge 3) 4 [293] 293 innerTail59_22 best = best := by
  rw [innerTail59_22, scan, if_neg (by decide), innerCandidates59_22_checked]
  change scan edge (search edge 3) 4 [293] 293 innerTail59_23
    (if 3 ≤ innerCandidates59_22.length then search edge 3 [293, 8009] innerCandidates59_22 8302 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan59_23

def innerTail59_21 : List Nat := 7757 :: innerTail59_22
def innerCandidates59_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates59_21_checked : candidates edge 4 293 7757 innerTail59_22 best.bound = innerCandidates59_21 := by
  unfold candidates
  have row : edge 7757 = SearchData.row7757.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan59_21 : scan edge (search edge 3) 4 [293] 293 innerTail59_21 best = best := by
  rw [innerTail59_21, scan, if_neg (by decide), innerCandidates59_21_checked]
  change scan edge (search edge 3) 4 [293] 293 innerTail59_22
    (if 3 ≤ innerCandidates59_21.length then search edge 3 [293, 7757] innerCandidates59_21 8050 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan59_22

def innerTail59_20 : List Nat := 7523 :: innerTail59_21
def innerCandidates59_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates59_20_checked : candidates edge 4 293 7523 innerTail59_21 best.bound = innerCandidates59_20 := by
  unfold candidates
  have row : edge 7523 = SearchData.row7523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan59_20 : scan edge (search edge 3) 4 [293] 293 innerTail59_20 best = best := by
  rw [innerTail59_20, scan, if_neg (by decide), innerCandidates59_20_checked]
  change scan edge (search edge 3) 4 [293] 293 innerTail59_21
    (if 3 ≤ innerCandidates59_20.length then search edge 3 [293, 7523] innerCandidates59_20 7816 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan59_21

def innerTail59_19 : List Nat := 5717 :: innerTail59_20
def innerCandidates59_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates59_19_checked : candidates edge 4 293 5717 innerTail59_20 best.bound = innerCandidates59_19 := by
  unfold candidates
  have row : edge 5717 = SearchData.row5717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan59_19 : scan edge (search edge 3) 4 [293] 293 innerTail59_19 best = best := by
  rw [innerTail59_19, scan, if_neg (by decide), innerCandidates59_19_checked]
  change scan edge (search edge 3) 4 [293] 293 innerTail59_20
    (if 3 ≤ innerCandidates59_19.length then search edge 3 [293, 5717] innerCandidates59_19 6010 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan59_20

def innerTail59_18 : List Nat := 5189 :: innerTail59_19
def innerCandidates59_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates59_18_checked : candidates edge 4 293 5189 innerTail59_19 best.bound = innerCandidates59_18 := by
  unfold candidates
  have row : edge 5189 = SearchData.row5189.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan59_18 : scan edge (search edge 3) 4 [293] 293 innerTail59_18 best = best := by
  rw [innerTail59_18, scan, if_neg (by decide), innerCandidates59_18_checked]
  change scan edge (search edge 3) 4 [293] 293 innerTail59_19
    (if 3 ≤ innerCandidates59_18.length then search edge 3 [293, 5189] innerCandidates59_18 5482 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan59_19

def innerTail59_17 : List Nat := 5171 :: innerTail59_18
def innerCandidates59_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates59_17_checked : candidates edge 4 293 5171 innerTail59_18 best.bound = innerCandidates59_17 := by
  unfold candidates
  have row : edge 5171 = SearchData.row5171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan59_17 : scan edge (search edge 3) 4 [293] 293 innerTail59_17 best = best := by
  rw [innerTail59_17, scan, if_neg (by decide), innerCandidates59_17_checked]
  change scan edge (search edge 3) 4 [293] 293 innerTail59_18
    (if 3 ≤ innerCandidates59_17.length then search edge 3 [293, 5171] innerCandidates59_17 5464 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan59_18

def innerTail59_16 : List Nat := 5147 :: innerTail59_17
def innerCandidates59_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates59_16_checked : candidates edge 4 293 5147 innerTail59_17 best.bound = innerCandidates59_16 := by
  unfold candidates
  have row : edge 5147 = SearchData.row5147.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan59_16 : scan edge (search edge 3) 4 [293] 293 innerTail59_16 best = best := by
  rw [innerTail59_16, scan, if_neg (by decide), innerCandidates59_16_checked]
  change scan edge (search edge 3) 4 [293] 293 innerTail59_17
    (if 3 ≤ innerCandidates59_16.length then search edge 3 [293, 5147] innerCandidates59_16 5440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan59_17

def innerTail59_15 : List Nat := 4751 :: innerTail59_16
def innerCandidates59_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates59_15_checked : candidates edge 4 293 4751 innerTail59_16 best.bound = innerCandidates59_15 := by
  unfold candidates
  have row : edge 4751 = SearchData.row4751.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan59_15 : scan edge (search edge 3) 4 [293] 293 innerTail59_15 best = best := by
  rw [innerTail59_15, scan, if_neg (by decide), innerCandidates59_15_checked]
  change scan edge (search edge 3) 4 [293] 293 innerTail59_16
    (if 3 ≤ innerCandidates59_15.length then search edge 3 [293, 4751] innerCandidates59_15 5044 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan59_16

def innerTail59_14 : List Nat := 4091 :: innerTail59_15
def innerCandidates59_14 : List Nat := [8237]
set_option maxRecDepth 4096 in
theorem innerCandidates59_14_checked : candidates edge 4 293 4091 innerTail59_15 best.bound = innerCandidates59_14 := by
  unfold candidates
  have row : edge 4091 = SearchData.row4091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan59_14 : scan edge (search edge 3) 4 [293] 293 innerTail59_14 best = best := by
  rw [innerTail59_14, scan, if_neg (by decide), innerCandidates59_14_checked]
  change scan edge (search edge 3) 4 [293] 293 innerTail59_15
    (if 3 ≤ innerCandidates59_14.length then search edge 3 [293, 4091] innerCandidates59_14 4384 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan59_15

def innerTail59_13 : List Nat := 3083 :: innerTail59_14
def innerCandidates59_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates59_13_checked : candidates edge 4 293 3083 innerTail59_14 best.bound = innerCandidates59_13 := by
  unfold candidates
  have row : edge 3083 = SearchData.row3083.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan59_13 : scan edge (search edge 3) 4 [293] 293 innerTail59_13 best = best := by
  rw [innerTail59_13, scan, if_neg (by decide), innerCandidates59_13_checked]
  change scan edge (search edge 3) 4 [293] 293 innerTail59_14
    (if 3 ≤ innerCandidates59_13.length then search edge 3 [293, 3083] innerCandidates59_13 3376 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan59_14

def innerTail59_12 : List Nat := 2777 :: innerTail59_13
def innerCandidates59_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates59_12_checked : candidates edge 4 293 2777 innerTail59_13 best.bound = innerCandidates59_12 := by
  unfold candidates
  have row : edge 2777 = SearchData.row2777.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan59_12 : scan edge (search edge 3) 4 [293] 293 innerTail59_12 best = best := by
  rw [innerTail59_12, scan, if_neg (by decide), innerCandidates59_12_checked]
  change scan edge (search edge 3) 4 [293] 293 innerTail59_13
    (if 3 ≤ innerCandidates59_12.length then search edge 3 [293, 2777] innerCandidates59_12 3070 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan59_13

def innerTail59_11 : List Nat := 2459 :: innerTail59_12
def innerCandidates59_11 : List Nat := [5171, 8009]
set_option maxRecDepth 4096 in
theorem innerCandidates59_11_checked : candidates edge 4 293 2459 innerTail59_12 best.bound = innerCandidates59_11 := by
  unfold candidates
  have row : edge 2459 = PrimePairCache.smallRow2459.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan59_11 : scan edge (search edge 3) 4 [293] 293 innerTail59_11 best = best := by
  rw [innerTail59_11, scan, if_neg (by decide), innerCandidates59_11_checked]
  change scan edge (search edge 3) 4 [293] 293 innerTail59_12
    (if 3 ≤ innerCandidates59_11.length then search edge 3 [293, 2459] innerCandidates59_11 2752 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan59_12

def innerTail59_10 : List Nat := 2411 :: innerTail59_11
def innerCandidates59_10 : List Nat := [5171]
set_option maxRecDepth 4096 in
theorem innerCandidates59_10_checked : candidates edge 4 293 2411 innerTail59_11 best.bound = innerCandidates59_10 := by
  unfold candidates
  have row : edge 2411 = PrimePairCache.smallRow2411.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan59_10 : scan edge (search edge 3) 4 [293] 293 innerTail59_10 best = best := by
  rw [innerTail59_10, scan, if_neg (by decide), innerCandidates59_10_checked]
  change scan edge (search edge 3) 4 [293] 293 innerTail59_11
    (if 3 ≤ innerCandidates59_10.length then search edge 3 [293, 2411] innerCandidates59_10 2704 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan59_11

def innerTail59_9 : List Nat := 1997 :: innerTail59_10
def innerCandidates59_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates59_9_checked : candidates edge 4 293 1997 innerTail59_10 best.bound = innerCandidates59_9 := by
  unfold candidates
  have row : edge 1997 = PrimePairCache.smallRow1997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan59_9 : scan edge (search edge 3) 4 [293] 293 innerTail59_9 best = best := by
  rw [innerTail59_9, scan, if_neg (by decide), innerCandidates59_9_checked]
  change scan edge (search edge 3) 4 [293] 293 innerTail59_10
    (if 3 ≤ innerCandidates59_9.length then search edge 3 [293, 1997] innerCandidates59_9 2290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan59_10

def innerTail59_8 : List Nat := 1823 :: innerTail59_9
def innerCandidates59_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates59_8_checked : candidates edge 4 293 1823 innerTail59_9 best.bound = innerCandidates59_8 := by
  unfold candidates
  have row : edge 1823 = PrimePairCache.smallRow1823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan59_8 : scan edge (search edge 3) 4 [293] 293 innerTail59_8 best = best := by
  rw [innerTail59_8, scan, if_neg (by decide), innerCandidates59_8_checked]
  change scan edge (search edge 3) 4 [293] 293 innerTail59_9
    (if 3 ≤ innerCandidates59_8.length then search edge 3 [293, 1823] innerCandidates59_8 2116 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan59_9

def innerTail59_7 : List Nat := 1583 :: innerTail59_8
def innerCandidates59_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates59_7_checked : candidates edge 4 293 1583 innerTail59_8 best.bound = innerCandidates59_7 := by
  unfold candidates
  have row : edge 1583 = PrimePairCache.smallRow1583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan59_7 : scan edge (search edge 3) 4 [293] 293 innerTail59_7 best = best := by
  rw [innerTail59_7, scan, if_neg (by decide), innerCandidates59_7_checked]
  change scan edge (search edge 3) 4 [293] 293 innerTail59_8
    (if 3 ≤ innerCandidates59_7.length then search edge 3 [293, 1583] innerCandidates59_7 1876 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan59_8

def innerTail59_6 : List Nat := 1493 :: innerTail59_7
def innerCandidates59_6 : List Nat := [5189]
set_option maxRecDepth 4096 in
theorem innerCandidates59_6_checked : candidates edge 4 293 1493 innerTail59_7 best.bound = innerCandidates59_6 := by
  unfold candidates
  have row : edge 1493 = PrimePairCache.smallRow1493.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan59_6 : scan edge (search edge 3) 4 [293] 293 innerTail59_6 best = best := by
  rw [innerTail59_6, scan, if_neg (by decide), innerCandidates59_6_checked]
  change scan edge (search edge 3) 4 [293] 293 innerTail59_7
    (if 3 ≤ innerCandidates59_6.length then search edge 3 [293, 1493] innerCandidates59_6 1786 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan59_7

def innerTail59_5 : List Nat := 1433 :: innerTail59_6
def innerCandidates59_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates59_5_checked : candidates edge 4 293 1433 innerTail59_6 best.bound = innerCandidates59_5 := by
  unfold candidates
  have row : edge 1433 = PrimePairCache.smallRow1433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan59_5 : scan edge (search edge 3) 4 [293] 293 innerTail59_5 best = best := by
  rw [innerTail59_5, scan, if_neg (by decide), innerCandidates59_5_checked]
  change scan edge (search edge 3) 4 [293] 293 innerTail59_6
    (if 3 ≤ innerCandidates59_5.length then search edge 3 [293, 1433] innerCandidates59_5 1726 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan59_6

def innerTail59_4 : List Nat := 1319 :: innerTail59_5
def innerCandidates59_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates59_4_checked : candidates edge 4 293 1319 innerTail59_5 best.bound = innerCandidates59_4 := by
  unfold candidates
  have row : edge 1319 = PrimePairCache.smallRow1319.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan59_4 : scan edge (search edge 3) 4 [293] 293 innerTail59_4 best = best := by
  rw [innerTail59_4, scan, if_neg (by decide), innerCandidates59_4_checked]
  change scan edge (search edge 3) 4 [293] 293 innerTail59_5
    (if 3 ≤ innerCandidates59_4.length then search edge 3 [293, 1319] innerCandidates59_4 1612 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan59_5

def innerTail59_3 : List Nat := 827 :: innerTail59_4
def innerCandidates59_3 : List Nat := [5717]
set_option maxRecDepth 4096 in
theorem innerCandidates59_3_checked : candidates edge 4 293 827 innerTail59_4 best.bound = innerCandidates59_3 := by
  unfold candidates
  have row : edge 827 = PrimePairCache.smallRow827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan59_3 : scan edge (search edge 3) 4 [293] 293 innerTail59_3 best = best := by
  rw [innerTail59_3, scan, if_neg (by decide), innerCandidates59_3_checked]
  change scan edge (search edge 3) 4 [293] 293 innerTail59_4
    (if 3 ≤ innerCandidates59_3.length then search edge 3 [293, 827] innerCandidates59_3 1120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan59_4

def innerTail59_2 : List Nat := 617 :: innerTail59_3
def innerCandidates59_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates59_2_checked : candidates edge 4 293 617 innerTail59_3 best.bound = innerCandidates59_2 := by
  unfold candidates
  have row : edge 617 = PrimePairCache.smallRow617.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan59_2 : scan edge (search edge 3) 4 [293] 293 innerTail59_2 best = best := by
  rw [innerTail59_2, scan, if_neg (by decide), innerCandidates59_2_checked]
  change scan edge (search edge 3) 4 [293] 293 innerTail59_3
    (if 3 ≤ innerCandidates59_2.length then search edge 3 [293, 617] innerCandidates59_2 910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan59_3

def innerTail59_1 : List Nat := 467 :: innerTail59_2
def innerCandidates59_1 : List Nat := [617]
set_option maxRecDepth 4096 in
theorem innerCandidates59_1_checked : candidates edge 4 293 467 innerTail59_2 best.bound = innerCandidates59_1 := by
  unfold candidates
  have row : edge 467 = PrimePairCache.smallRow467.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan59_1 : scan edge (search edge 3) 4 [293] 293 innerTail59_1 best = best := by
  rw [innerTail59_1, scan, if_neg (by decide), innerCandidates59_1_checked]
  change scan edge (search edge 3) 4 [293] 293 innerTail59_2
    (if 3 ≤ innerCandidates59_1.length then search edge 3 [293, 467] innerCandidates59_1 760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan59_2

def innerTail59_0 : List Nat := 311 :: innerTail59_1
def innerCandidates59_0 : List Nat := [827, 5171, 7523, 8237]
set_option maxRecDepth 4096 in
theorem innerCandidates59_0_checked : candidates edge 4 293 311 innerTail59_1 best.bound = innerCandidates59_0 := by
  unfold candidates
  have row : edge 311 = PrimePairCache.smallRow311.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild59_0 : search edge 3 [293, 311] innerCandidates59_0 604 best = best := by decide

theorem innerScan59_0 : scan edge (search edge 3) 4 [293] 293 innerTail59_0 best = best := by
  rw [innerTail59_0, scan, if_neg (by decide), innerCandidates59_0_checked]
  change scan edge (search edge 3) 4 [293] 293 innerTail59_1
    (if 3 ≤ innerCandidates59_0.length then search edge 3 [293, 311] innerCandidates59_0 604 best else best) = best
  rw [if_pos (by decide), innerChild59_0]
  exact innerScan59_1

theorem rootBranch59 : search edge 4 [293] rootCandidates59 293 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [293] 293 innerTail59_0 best = best
  exact innerScan59_0

end ProjectEulerSolutions.PrimePairUpper.Search
