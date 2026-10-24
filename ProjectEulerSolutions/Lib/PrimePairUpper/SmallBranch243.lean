import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates25 : List Nat := [449, 857, 1061, 1193, 1439, 1601, 1907, 2129, 2339, 2381, 2633, 2843, 2957, 2999, 3221, 4289, 4523, 4691, 4919, 4973, 5279, 5507, 5531, 5897, 6359, 6563, 6917, 7607]
set_option maxRecDepth 4096 in
theorem rootCandidates25_checked : candidates edge 5 0 107 tail26 best.bound = rootCandidates25 := by
  unfold candidates
  have row : edge 107 = PrimePairCache.smallRow107.contains := by funext q; rfl
  rw [row]
  decide

def innerTail25_28 : List Nat := []
theorem innerScan25_28 : scan edge (search edge 3) 4 [107] 107 innerTail25_28 best = best := rfl

def innerTail25_27 : List Nat := 7607 :: innerTail25_28
def innerCandidates25_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates25_27_checked : candidates edge 4 107 7607 innerTail25_28 best.bound = innerCandidates25_27 := by
  unfold candidates
  have row : edge 7607 = SearchData.row7607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan25_27 : scan edge (search edge 3) 4 [107] 107 innerTail25_27 best = best := by
  rw [innerTail25_27, scan, if_neg (by decide), innerCandidates25_27_checked]
  change scan edge (search edge 3) 4 [107] 107 innerTail25_28
    (if 3 ≤ innerCandidates25_27.length then search edge 3 [107, 7607] innerCandidates25_27 7714 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan25_28

def innerTail25_26 : List Nat := 6917 :: innerTail25_27
def innerCandidates25_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates25_26_checked : candidates edge 4 107 6917 innerTail25_27 best.bound = innerCandidates25_26 := by
  unfold candidates
  have row : edge 6917 = SearchData.row6917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan25_26 : scan edge (search edge 3) 4 [107] 107 innerTail25_26 best = best := by
  rw [innerTail25_26, scan, if_neg (by decide), innerCandidates25_26_checked]
  change scan edge (search edge 3) 4 [107] 107 innerTail25_27
    (if 3 ≤ innerCandidates25_26.length then search edge 3 [107, 6917] innerCandidates25_26 7024 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan25_27

def innerTail25_25 : List Nat := 6563 :: innerTail25_26
def innerCandidates25_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates25_25_checked : candidates edge 4 107 6563 innerTail25_26 best.bound = innerCandidates25_25 := by
  unfold candidates
  have row : edge 6563 = SearchData.row6563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan25_25 : scan edge (search edge 3) 4 [107] 107 innerTail25_25 best = best := by
  rw [innerTail25_25, scan, if_neg (by decide), innerCandidates25_25_checked]
  change scan edge (search edge 3) 4 [107] 107 innerTail25_26
    (if 3 ≤ innerCandidates25_25.length then search edge 3 [107, 6563] innerCandidates25_25 6670 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan25_26

def innerTail25_24 : List Nat := 6359 :: innerTail25_25
def innerCandidates25_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates25_24_checked : candidates edge 4 107 6359 innerTail25_25 best.bound = innerCandidates25_24 := by
  unfold candidates
  have row : edge 6359 = SearchData.row6359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan25_24 : scan edge (search edge 3) 4 [107] 107 innerTail25_24 best = best := by
  rw [innerTail25_24, scan, if_neg (by decide), innerCandidates25_24_checked]
  change scan edge (search edge 3) 4 [107] 107 innerTail25_25
    (if 3 ≤ innerCandidates25_24.length then search edge 3 [107, 6359] innerCandidates25_24 6466 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan25_25

def innerTail25_23 : List Nat := 5897 :: innerTail25_24
def innerCandidates25_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates25_23_checked : candidates edge 4 107 5897 innerTail25_24 best.bound = innerCandidates25_23 := by
  unfold candidates
  have row : edge 5897 = SearchData.row5897.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan25_23 : scan edge (search edge 3) 4 [107] 107 innerTail25_23 best = best := by
  rw [innerTail25_23, scan, if_neg (by decide), innerCandidates25_23_checked]
  change scan edge (search edge 3) 4 [107] 107 innerTail25_24
    (if 3 ≤ innerCandidates25_23.length then search edge 3 [107, 5897] innerCandidates25_23 6004 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan25_24

def innerTail25_22 : List Nat := 5531 :: innerTail25_23
def innerCandidates25_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates25_22_checked : candidates edge 4 107 5531 innerTail25_23 best.bound = innerCandidates25_22 := by
  unfold candidates
  have row : edge 5531 = SearchData.row5531.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan25_22 : scan edge (search edge 3) 4 [107] 107 innerTail25_22 best = best := by
  rw [innerTail25_22, scan, if_neg (by decide), innerCandidates25_22_checked]
  change scan edge (search edge 3) 4 [107] 107 innerTail25_23
    (if 3 ≤ innerCandidates25_22.length then search edge 3 [107, 5531] innerCandidates25_22 5638 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan25_23

def innerTail25_21 : List Nat := 5507 :: innerTail25_22
def innerCandidates25_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates25_21_checked : candidates edge 4 107 5507 innerTail25_22 best.bound = innerCandidates25_21 := by
  unfold candidates
  have row : edge 5507 = SearchData.row5507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan25_21 : scan edge (search edge 3) 4 [107] 107 innerTail25_21 best = best := by
  rw [innerTail25_21, scan, if_neg (by decide), innerCandidates25_21_checked]
  change scan edge (search edge 3) 4 [107] 107 innerTail25_22
    (if 3 ≤ innerCandidates25_21.length then search edge 3 [107, 5507] innerCandidates25_21 5614 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan25_22

def innerTail25_20 : List Nat := 5279 :: innerTail25_21
def innerCandidates25_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates25_20_checked : candidates edge 4 107 5279 innerTail25_21 best.bound = innerCandidates25_20 := by
  unfold candidates
  have row : edge 5279 = SearchData.row5279.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan25_20 : scan edge (search edge 3) 4 [107] 107 innerTail25_20 best = best := by
  rw [innerTail25_20, scan, if_neg (by decide), innerCandidates25_20_checked]
  change scan edge (search edge 3) 4 [107] 107 innerTail25_21
    (if 3 ≤ innerCandidates25_20.length then search edge 3 [107, 5279] innerCandidates25_20 5386 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan25_21

def innerTail25_19 : List Nat := 4973 :: innerTail25_20
def innerCandidates25_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates25_19_checked : candidates edge 4 107 4973 innerTail25_20 best.bound = innerCandidates25_19 := by
  unfold candidates
  have row : edge 4973 = SearchData.row4973.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan25_19 : scan edge (search edge 3) 4 [107] 107 innerTail25_19 best = best := by
  rw [innerTail25_19, scan, if_neg (by decide), innerCandidates25_19_checked]
  change scan edge (search edge 3) 4 [107] 107 innerTail25_20
    (if 3 ≤ innerCandidates25_19.length then search edge 3 [107, 4973] innerCandidates25_19 5080 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan25_20

def innerTail25_18 : List Nat := 4919 :: innerTail25_19
def innerCandidates25_18 : List Nat := [5897, 7607]
set_option maxRecDepth 4096 in
theorem innerCandidates25_18_checked : candidates edge 4 107 4919 innerTail25_19 best.bound = innerCandidates25_18 := by
  unfold candidates
  have row : edge 4919 = SearchData.row4919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan25_18 : scan edge (search edge 3) 4 [107] 107 innerTail25_18 best = best := by
  rw [innerTail25_18, scan, if_neg (by decide), innerCandidates25_18_checked]
  change scan edge (search edge 3) 4 [107] 107 innerTail25_19
    (if 3 ≤ innerCandidates25_18.length then search edge 3 [107, 4919] innerCandidates25_18 5026 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan25_19

def innerTail25_17 : List Nat := 4691 :: innerTail25_18
def innerCandidates25_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates25_17_checked : candidates edge 4 107 4691 innerTail25_18 best.bound = innerCandidates25_17 := by
  unfold candidates
  have row : edge 4691 = SearchData.row4691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan25_17 : scan edge (search edge 3) 4 [107] 107 innerTail25_17 best = best := by
  rw [innerTail25_17, scan, if_neg (by decide), innerCandidates25_17_checked]
  change scan edge (search edge 3) 4 [107] 107 innerTail25_18
    (if 3 ≤ innerCandidates25_17.length then search edge 3 [107, 4691] innerCandidates25_17 4798 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan25_18

def innerTail25_16 : List Nat := 4523 :: innerTail25_17
def innerCandidates25_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates25_16_checked : candidates edge 4 107 4523 innerTail25_17 best.bound = innerCandidates25_16 := by
  unfold candidates
  have row : edge 4523 = SearchData.row4523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan25_16 : scan edge (search edge 3) 4 [107] 107 innerTail25_16 best = best := by
  rw [innerTail25_16, scan, if_neg (by decide), innerCandidates25_16_checked]
  change scan edge (search edge 3) 4 [107] 107 innerTail25_17
    (if 3 ≤ innerCandidates25_16.length then search edge 3 [107, 4523] innerCandidates25_16 4630 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan25_17

def innerTail25_15 : List Nat := 4289 :: innerTail25_16
def innerCandidates25_15 : List Nat := [5507]
set_option maxRecDepth 4096 in
theorem innerCandidates25_15_checked : candidates edge 4 107 4289 innerTail25_16 best.bound = innerCandidates25_15 := by
  unfold candidates
  have row : edge 4289 = SearchData.row4289.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan25_15 : scan edge (search edge 3) 4 [107] 107 innerTail25_15 best = best := by
  rw [innerTail25_15, scan, if_neg (by decide), innerCandidates25_15_checked]
  change scan edge (search edge 3) 4 [107] 107 innerTail25_16
    (if 3 ≤ innerCandidates25_15.length then search edge 3 [107, 4289] innerCandidates25_15 4396 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan25_16

def innerTail25_14 : List Nat := 3221 :: innerTail25_15
def innerCandidates25_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates25_14_checked : candidates edge 4 107 3221 innerTail25_15 best.bound = innerCandidates25_14 := by
  unfold candidates
  have row : edge 3221 = SearchData.row3221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan25_14 : scan edge (search edge 3) 4 [107] 107 innerTail25_14 best = best := by
  rw [innerTail25_14, scan, if_neg (by decide), innerCandidates25_14_checked]
  change scan edge (search edge 3) 4 [107] 107 innerTail25_15
    (if 3 ≤ innerCandidates25_14.length then search edge 3 [107, 3221] innerCandidates25_14 3328 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan25_15

def innerTail25_13 : List Nat := 2999 :: innerTail25_14
def innerCandidates25_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates25_13_checked : candidates edge 4 107 2999 innerTail25_14 best.bound = innerCandidates25_13 := by
  unfold candidates
  have row : edge 2999 = SearchData.row2999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan25_13 : scan edge (search edge 3) 4 [107] 107 innerTail25_13 best = best := by
  rw [innerTail25_13, scan, if_neg (by decide), innerCandidates25_13_checked]
  change scan edge (search edge 3) 4 [107] 107 innerTail25_14
    (if 3 ≤ innerCandidates25_13.length then search edge 3 [107, 2999] innerCandidates25_13 3106 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan25_14

def innerTail25_12 : List Nat := 2957 :: innerTail25_13
def innerCandidates25_12 : List Nat := [4973]
set_option maxRecDepth 4096 in
theorem innerCandidates25_12_checked : candidates edge 4 107 2957 innerTail25_13 best.bound = innerCandidates25_12 := by
  unfold candidates
  have row : edge 2957 = SearchData.row2957.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan25_12 : scan edge (search edge 3) 4 [107] 107 innerTail25_12 best = best := by
  rw [innerTail25_12, scan, if_neg (by decide), innerCandidates25_12_checked]
  change scan edge (search edge 3) 4 [107] 107 innerTail25_13
    (if 3 ≤ innerCandidates25_12.length then search edge 3 [107, 2957] innerCandidates25_12 3064 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan25_13

def innerTail25_11 : List Nat := 2843 :: innerTail25_12
def innerCandidates25_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates25_11_checked : candidates edge 4 107 2843 innerTail25_12 best.bound = innerCandidates25_11 := by
  unfold candidates
  have row : edge 2843 = SearchData.row2843.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan25_11 : scan edge (search edge 3) 4 [107] 107 innerTail25_11 best = best := by
  rw [innerTail25_11, scan, if_neg (by decide), innerCandidates25_11_checked]
  change scan edge (search edge 3) 4 [107] 107 innerTail25_12
    (if 3 ≤ innerCandidates25_11.length then search edge 3 [107, 2843] innerCandidates25_11 2950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan25_12

def innerTail25_10 : List Nat := 2633 :: innerTail25_11
def innerCandidates25_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates25_10_checked : candidates edge 4 107 2633 innerTail25_11 best.bound = innerCandidates25_10 := by
  unfold candidates
  have row : edge 2633 = PrimePairCache.smallRow2633.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan25_10 : scan edge (search edge 3) 4 [107] 107 innerTail25_10 best = best := by
  rw [innerTail25_10, scan, if_neg (by decide), innerCandidates25_10_checked]
  change scan edge (search edge 3) 4 [107] 107 innerTail25_11
    (if 3 ≤ innerCandidates25_10.length then search edge 3 [107, 2633] innerCandidates25_10 2740 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan25_11

def innerTail25_9 : List Nat := 2381 :: innerTail25_10
def innerCandidates25_9 : List Nat := [3221, 6563]
set_option maxRecDepth 4096 in
theorem innerCandidates25_9_checked : candidates edge 4 107 2381 innerTail25_10 best.bound = innerCandidates25_9 := by
  unfold candidates
  have row : edge 2381 = PrimePairCache.smallRow2381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan25_9 : scan edge (search edge 3) 4 [107] 107 innerTail25_9 best = best := by
  rw [innerTail25_9, scan, if_neg (by decide), innerCandidates25_9_checked]
  change scan edge (search edge 3) 4 [107] 107 innerTail25_10
    (if 3 ≤ innerCandidates25_9.length then search edge 3 [107, 2381] innerCandidates25_9 2488 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan25_10

def innerTail25_8 : List Nat := 2339 :: innerTail25_9
def innerCandidates25_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates25_8_checked : candidates edge 4 107 2339 innerTail25_9 best.bound = innerCandidates25_8 := by
  unfold candidates
  have row : edge 2339 = PrimePairCache.smallRow2339.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan25_8 : scan edge (search edge 3) 4 [107] 107 innerTail25_8 best = best := by
  rw [innerTail25_8, scan, if_neg (by decide), innerCandidates25_8_checked]
  change scan edge (search edge 3) 4 [107] 107 innerTail25_9
    (if 3 ≤ innerCandidates25_8.length then search edge 3 [107, 2339] innerCandidates25_8 2446 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan25_9

def innerTail25_7 : List Nat := 2129 :: innerTail25_8
def innerCandidates25_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates25_7_checked : candidates edge 4 107 2129 innerTail25_8 best.bound = innerCandidates25_7 := by
  unfold candidates
  have row : edge 2129 = PrimePairCache.smallRow2129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan25_7 : scan edge (search edge 3) 4 [107] 107 innerTail25_7 best = best := by
  rw [innerTail25_7, scan, if_neg (by decide), innerCandidates25_7_checked]
  change scan edge (search edge 3) 4 [107] 107 innerTail25_8
    (if 3 ≤ innerCandidates25_7.length then search edge 3 [107, 2129] innerCandidates25_7 2236 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan25_8

def innerTail25_6 : List Nat := 1907 :: innerTail25_7
def innerCandidates25_6 : List Nat := [5897]
set_option maxRecDepth 4096 in
theorem innerCandidates25_6_checked : candidates edge 4 107 1907 innerTail25_7 best.bound = innerCandidates25_6 := by
  unfold candidates
  have row : edge 1907 = PrimePairCache.smallRow1907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan25_6 : scan edge (search edge 3) 4 [107] 107 innerTail25_6 best = best := by
  rw [innerTail25_6, scan, if_neg (by decide), innerCandidates25_6_checked]
  change scan edge (search edge 3) 4 [107] 107 innerTail25_7
    (if 3 ≤ innerCandidates25_6.length then search edge 3 [107, 1907] innerCandidates25_6 2014 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan25_7

def innerTail25_5 : List Nat := 1601 :: innerTail25_6
def innerCandidates25_5 : List Nat := [6359]
set_option maxRecDepth 4096 in
theorem innerCandidates25_5_checked : candidates edge 4 107 1601 innerTail25_6 best.bound = innerCandidates25_5 := by
  unfold candidates
  have row : edge 1601 = PrimePairCache.smallRow1601.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan25_5 : scan edge (search edge 3) 4 [107] 107 innerTail25_5 best = best := by
  rw [innerTail25_5, scan, if_neg (by decide), innerCandidates25_5_checked]
  change scan edge (search edge 3) 4 [107] 107 innerTail25_6
    (if 3 ≤ innerCandidates25_5.length then search edge 3 [107, 1601] innerCandidates25_5 1708 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan25_6

def innerTail25_4 : List Nat := 1439 :: innerTail25_5
def innerCandidates25_4 : List Nat := [5279]
set_option maxRecDepth 4096 in
theorem innerCandidates25_4_checked : candidates edge 4 107 1439 innerTail25_5 best.bound = innerCandidates25_4 := by
  unfold candidates
  have row : edge 1439 = PrimePairCache.smallRow1439.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan25_4 : scan edge (search edge 3) 4 [107] 107 innerTail25_4 best = best := by
  rw [innerTail25_4, scan, if_neg (by decide), innerCandidates25_4_checked]
  change scan edge (search edge 3) 4 [107] 107 innerTail25_5
    (if 3 ≤ innerCandidates25_4.length then search edge 3 [107, 1439] innerCandidates25_4 1546 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan25_5

def innerTail25_3 : List Nat := 1193 :: innerTail25_4
def innerCandidates25_3 : List Nat := [5507]
set_option maxRecDepth 4096 in
theorem innerCandidates25_3_checked : candidates edge 4 107 1193 innerTail25_4 best.bound = innerCandidates25_3 := by
  unfold candidates
  have row : edge 1193 = PrimePairCache.smallRow1193.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan25_3 : scan edge (search edge 3) 4 [107] 107 innerTail25_3 best = best := by
  rw [innerTail25_3, scan, if_neg (by decide), innerCandidates25_3_checked]
  change scan edge (search edge 3) 4 [107] 107 innerTail25_4
    (if 3 ≤ innerCandidates25_3.length then search edge 3 [107, 1193] innerCandidates25_3 1300 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan25_4

def innerTail25_2 : List Nat := 1061 :: innerTail25_3
def innerCandidates25_2 : List Nat := [4973]
set_option maxRecDepth 4096 in
theorem innerCandidates25_2_checked : candidates edge 4 107 1061 innerTail25_3 best.bound = innerCandidates25_2 := by
  unfold candidates
  have row : edge 1061 = PrimePairCache.smallRow1061.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan25_2 : scan edge (search edge 3) 4 [107] 107 innerTail25_2 best = best := by
  rw [innerTail25_2, scan, if_neg (by decide), innerCandidates25_2_checked]
  change scan edge (search edge 3) 4 [107] 107 innerTail25_3
    (if 3 ≤ innerCandidates25_2.length then search edge 3 [107, 1061] innerCandidates25_2 1168 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan25_3

def innerTail25_1 : List Nat := 857 :: innerTail25_2
def innerCandidates25_1 : List Nat := [1601, 2633]
set_option maxRecDepth 4096 in
theorem innerCandidates25_1_checked : candidates edge 4 107 857 innerTail25_2 best.bound = innerCandidates25_1 := by
  unfold candidates
  have row : edge 857 = PrimePairCache.smallRow857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan25_1 : scan edge (search edge 3) 4 [107] 107 innerTail25_1 best = best := by
  rw [innerTail25_1, scan, if_neg (by decide), innerCandidates25_1_checked]
  change scan edge (search edge 3) 4 [107] 107 innerTail25_2
    (if 3 ≤ innerCandidates25_1.length then search edge 3 [107, 857] innerCandidates25_1 964 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan25_2

def innerTail25_0 : List Nat := 449 :: innerTail25_1
def innerCandidates25_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates25_0_checked : candidates edge 4 107 449 innerTail25_1 best.bound = innerCandidates25_0 := by
  unfold candidates
  have row : edge 449 = PrimePairCache.smallRow449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan25_0 : scan edge (search edge 3) 4 [107] 107 innerTail25_0 best = best := by
  rw [innerTail25_0, scan, if_neg (by decide), innerCandidates25_0_checked]
  change scan edge (search edge 3) 4 [107] 107 innerTail25_1
    (if 3 ≤ innerCandidates25_0.length then search edge 3 [107, 449] innerCandidates25_0 556 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan25_1

theorem rootBranch25 : search edge 4 [107] rootCandidates25 107 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [107] 107 innerTail25_0 best = best
  exact innerScan25_0

def rootCandidates24 : List Nat := [307, 421, 997, 1483, 1669, 1759, 1831, 1999, 2341, 2467, 2617, 2683, 2803, 3061, 4027, 4339, 4357, 4567, 5413, 5527, 5869, 6229, 6247, 6661, 7489, 7879, 8269, 8329]
set_option maxRecDepth 4096 in
theorem rootCandidates24_checked : candidates edge 5 0 103 tail25 best.bound = rootCandidates24 := by
  unfold candidates
  have row : edge 103 = PrimePairCache.smallRow103.contains := by funext q; rfl
  rw [row]
  decide

def innerTail24_28 : List Nat := []
theorem innerScan24_28 : scan edge (search edge 3) 4 [103] 103 innerTail24_28 best = best := rfl

def innerTail24_27 : List Nat := 8329 :: innerTail24_28
def innerCandidates24_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates24_27_checked : candidates edge 4 103 8329 innerTail24_28 best.bound = innerCandidates24_27 := by
  unfold candidates
  have row : edge 8329 = SearchData.row8329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan24_27 : scan edge (search edge 3) 4 [103] 103 innerTail24_27 best = best := by
  rw [innerTail24_27, scan, if_neg (by decide), innerCandidates24_27_checked]
  change scan edge (search edge 3) 4 [103] 103 innerTail24_28
    (if 3 ≤ innerCandidates24_27.length then search edge 3 [103, 8329] innerCandidates24_27 8432 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan24_28

def innerTail24_26 : List Nat := 8269 :: innerTail24_27
def innerCandidates24_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates24_26_checked : candidates edge 4 103 8269 innerTail24_27 best.bound = innerCandidates24_26 := by
  unfold candidates
  have row : edge 8269 = SearchData.row8269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan24_26 : scan edge (search edge 3) 4 [103] 103 innerTail24_26 best = best := by
  rw [innerTail24_26, scan, if_neg (by decide), innerCandidates24_26_checked]
  change scan edge (search edge 3) 4 [103] 103 innerTail24_27
    (if 3 ≤ innerCandidates24_26.length then search edge 3 [103, 8269] innerCandidates24_26 8372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan24_27

def innerTail24_25 : List Nat := 7879 :: innerTail24_26
def innerCandidates24_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates24_25_checked : candidates edge 4 103 7879 innerTail24_26 best.bound = innerCandidates24_25 := by
  unfold candidates
  have row : edge 7879 = SearchData.row7879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan24_25 : scan edge (search edge 3) 4 [103] 103 innerTail24_25 best = best := by
  rw [innerTail24_25, scan, if_neg (by decide), innerCandidates24_25_checked]
  change scan edge (search edge 3) 4 [103] 103 innerTail24_26
    (if 3 ≤ innerCandidates24_25.length then search edge 3 [103, 7879] innerCandidates24_25 7982 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan24_26

def innerTail24_24 : List Nat := 7489 :: innerTail24_25
def innerCandidates24_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates24_24_checked : candidates edge 4 103 7489 innerTail24_25 best.bound = innerCandidates24_24 := by
  unfold candidates
  have row : edge 7489 = SearchData.row7489.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan24_24 : scan edge (search edge 3) 4 [103] 103 innerTail24_24 best = best := by
  rw [innerTail24_24, scan, if_neg (by decide), innerCandidates24_24_checked]
  change scan edge (search edge 3) 4 [103] 103 innerTail24_25
    (if 3 ≤ innerCandidates24_24.length then search edge 3 [103, 7489] innerCandidates24_24 7592 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan24_25

def innerTail24_23 : List Nat := 6661 :: innerTail24_24
def innerCandidates24_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates24_23_checked : candidates edge 4 103 6661 innerTail24_24 best.bound = innerCandidates24_23 := by
  unfold candidates
  have row : edge 6661 = SearchData.row6661.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan24_23 : scan edge (search edge 3) 4 [103] 103 innerTail24_23 best = best := by
  rw [innerTail24_23, scan, if_neg (by decide), innerCandidates24_23_checked]
  change scan edge (search edge 3) 4 [103] 103 innerTail24_24
    (if 3 ≤ innerCandidates24_23.length then search edge 3 [103, 6661] innerCandidates24_23 6764 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan24_24

def innerTail24_22 : List Nat := 6247 :: innerTail24_23
def innerCandidates24_22 : List Nat := [7879]
set_option maxRecDepth 4096 in
theorem innerCandidates24_22_checked : candidates edge 4 103 6247 innerTail24_23 best.bound = innerCandidates24_22 := by
  unfold candidates
  have row : edge 6247 = SearchData.row6247.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan24_22 : scan edge (search edge 3) 4 [103] 103 innerTail24_22 best = best := by
  rw [innerTail24_22, scan, if_neg (by decide), innerCandidates24_22_checked]
  change scan edge (search edge 3) 4 [103] 103 innerTail24_23
    (if 3 ≤ innerCandidates24_22.length then search edge 3 [103, 6247] innerCandidates24_22 6350 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan24_23

def innerTail24_21 : List Nat := 6229 :: innerTail24_22
def innerCandidates24_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates24_21_checked : candidates edge 4 103 6229 innerTail24_22 best.bound = innerCandidates24_21 := by
  unfold candidates
  have row : edge 6229 = SearchData.row6229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan24_21 : scan edge (search edge 3) 4 [103] 103 innerTail24_21 best = best := by
  rw [innerTail24_21, scan, if_neg (by decide), innerCandidates24_21_checked]
  change scan edge (search edge 3) 4 [103] 103 innerTail24_22
    (if 3 ≤ innerCandidates24_21.length then search edge 3 [103, 6229] innerCandidates24_21 6332 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan24_22

def innerTail24_20 : List Nat := 5869 :: innerTail24_21
def innerCandidates24_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates24_20_checked : candidates edge 4 103 5869 innerTail24_21 best.bound = innerCandidates24_20 := by
  unfold candidates
  have row : edge 5869 = SearchData.row5869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan24_20 : scan edge (search edge 3) 4 [103] 103 innerTail24_20 best = best := by
  rw [innerTail24_20, scan, if_neg (by decide), innerCandidates24_20_checked]
  change scan edge (search edge 3) 4 [103] 103 innerTail24_21
    (if 3 ≤ innerCandidates24_20.length then search edge 3 [103, 5869] innerCandidates24_20 5972 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan24_21

def innerTail24_19 : List Nat := 5527 :: innerTail24_20
def innerCandidates24_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates24_19_checked : candidates edge 4 103 5527 innerTail24_20 best.bound = innerCandidates24_19 := by
  unfold candidates
  have row : edge 5527 = SearchData.row5527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan24_19 : scan edge (search edge 3) 4 [103] 103 innerTail24_19 best = best := by
  rw [innerTail24_19, scan, if_neg (by decide), innerCandidates24_19_checked]
  change scan edge (search edge 3) 4 [103] 103 innerTail24_20
    (if 3 ≤ innerCandidates24_19.length then search edge 3 [103, 5527] innerCandidates24_19 5630 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan24_20

def innerTail24_18 : List Nat := 5413 :: innerTail24_19
def innerCandidates24_18 : List Nat := [5527]
set_option maxRecDepth 4096 in
theorem innerCandidates24_18_checked : candidates edge 4 103 5413 innerTail24_19 best.bound = innerCandidates24_18 := by
  unfold candidates
  have row : edge 5413 = SearchData.row5413.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan24_18 : scan edge (search edge 3) 4 [103] 103 innerTail24_18 best = best := by
  rw [innerTail24_18, scan, if_neg (by decide), innerCandidates24_18_checked]
  change scan edge (search edge 3) 4 [103] 103 innerTail24_19
    (if 3 ≤ innerCandidates24_18.length then search edge 3 [103, 5413] innerCandidates24_18 5516 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan24_19

def innerTail24_17 : List Nat := 4567 :: innerTail24_18
def innerCandidates24_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates24_17_checked : candidates edge 4 103 4567 innerTail24_18 best.bound = innerCandidates24_17 := by
  unfold candidates
  have row : edge 4567 = SearchData.row4567.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan24_17 : scan edge (search edge 3) 4 [103] 103 innerTail24_17 best = best := by
  rw [innerTail24_17, scan, if_neg (by decide), innerCandidates24_17_checked]
  change scan edge (search edge 3) 4 [103] 103 innerTail24_18
    (if 3 ≤ innerCandidates24_17.length then search edge 3 [103, 4567] innerCandidates24_17 4670 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan24_18

def innerTail24_16 : List Nat := 4357 :: innerTail24_17
def innerCandidates24_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates24_16_checked : candidates edge 4 103 4357 innerTail24_17 best.bound = innerCandidates24_16 := by
  unfold candidates
  have row : edge 4357 = SearchData.row4357.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan24_16 : scan edge (search edge 3) 4 [103] 103 innerTail24_16 best = best := by
  rw [innerTail24_16, scan, if_neg (by decide), innerCandidates24_16_checked]
  change scan edge (search edge 3) 4 [103] 103 innerTail24_17
    (if 3 ≤ innerCandidates24_16.length then search edge 3 [103, 4357] innerCandidates24_16 4460 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan24_17

def innerTail24_15 : List Nat := 4339 :: innerTail24_16
def innerCandidates24_15 : List Nat := [4567]
set_option maxRecDepth 4096 in
theorem innerCandidates24_15_checked : candidates edge 4 103 4339 innerTail24_16 best.bound = innerCandidates24_15 := by
  unfold candidates
  have row : edge 4339 = SearchData.row4339.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan24_15 : scan edge (search edge 3) 4 [103] 103 innerTail24_15 best = best := by
  rw [innerTail24_15, scan, if_neg (by decide), innerCandidates24_15_checked]
  change scan edge (search edge 3) 4 [103] 103 innerTail24_16
    (if 3 ≤ innerCandidates24_15.length then search edge 3 [103, 4339] innerCandidates24_15 4442 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan24_16

def innerTail24_14 : List Nat := 4027 :: innerTail24_15
def innerCandidates24_14 : List Nat := [5413]
set_option maxRecDepth 4096 in
theorem innerCandidates24_14_checked : candidates edge 4 103 4027 innerTail24_15 best.bound = innerCandidates24_14 := by
  unfold candidates
  have row : edge 4027 = SearchData.row4027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan24_14 : scan edge (search edge 3) 4 [103] 103 innerTail24_14 best = best := by
  rw [innerTail24_14, scan, if_neg (by decide), innerCandidates24_14_checked]
  change scan edge (search edge 3) 4 [103] 103 innerTail24_15
    (if 3 ≤ innerCandidates24_14.length then search edge 3 [103, 4027] innerCandidates24_14 4130 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan24_15

def innerTail24_13 : List Nat := 3061 :: innerTail24_14
def innerCandidates24_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates24_13_checked : candidates edge 4 103 3061 innerTail24_14 best.bound = innerCandidates24_13 := by
  unfold candidates
  have row : edge 3061 = SearchData.row3061.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan24_13 : scan edge (search edge 3) 4 [103] 103 innerTail24_13 best = best := by
  rw [innerTail24_13, scan, if_neg (by decide), innerCandidates24_13_checked]
  change scan edge (search edge 3) 4 [103] 103 innerTail24_14
    (if 3 ≤ innerCandidates24_13.length then search edge 3 [103, 3061] innerCandidates24_13 3164 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan24_14

def innerTail24_12 : List Nat := 2803 :: innerTail24_13
def innerCandidates24_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates24_12_checked : candidates edge 4 103 2803 innerTail24_13 best.bound = innerCandidates24_12 := by
  unfold candidates
  have row : edge 2803 = SearchData.row2803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan24_12 : scan edge (search edge 3) 4 [103] 103 innerTail24_12 best = best := by
  rw [innerTail24_12, scan, if_neg (by decide), innerCandidates24_12_checked]
  change scan edge (search edge 3) 4 [103] 103 innerTail24_13
    (if 3 ≤ innerCandidates24_12.length then search edge 3 [103, 2803] innerCandidates24_12 2906 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan24_13

def innerTail24_11 : List Nat := 2683 :: innerTail24_12
def innerCandidates24_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates24_11_checked : candidates edge 4 103 2683 innerTail24_12 best.bound = innerCandidates24_11 := by
  unfold candidates
  have row : edge 2683 = SearchData.row2683.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan24_11 : scan edge (search edge 3) 4 [103] 103 innerTail24_11 best = best := by
  rw [innerTail24_11, scan, if_neg (by decide), innerCandidates24_11_checked]
  change scan edge (search edge 3) 4 [103] 103 innerTail24_12
    (if 3 ≤ innerCandidates24_11.length then search edge 3 [103, 2683] innerCandidates24_11 2786 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan24_12

def innerTail24_10 : List Nat := 2617 :: innerTail24_11
def innerCandidates24_10 : List Nat := [2803, 4567]
set_option maxRecDepth 4096 in
theorem innerCandidates24_10_checked : candidates edge 4 103 2617 innerTail24_11 best.bound = innerCandidates24_10 := by
  unfold candidates
  have row : edge 2617 = PrimePairCache.smallRow2617.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan24_10 : scan edge (search edge 3) 4 [103] 103 innerTail24_10 best = best := by
  rw [innerTail24_10, scan, if_neg (by decide), innerCandidates24_10_checked]
  change scan edge (search edge 3) 4 [103] 103 innerTail24_11
    (if 3 ≤ innerCandidates24_10.length then search edge 3 [103, 2617] innerCandidates24_10 2720 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan24_11

def innerTail24_9 : List Nat := 2467 :: innerTail24_10
def innerCandidates24_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates24_9_checked : candidates edge 4 103 2467 innerTail24_10 best.bound = innerCandidates24_9 := by
  unfold candidates
  have row : edge 2467 = PrimePairCache.smallRow2467.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan24_9 : scan edge (search edge 3) 4 [103] 103 innerTail24_9 best = best := by
  rw [innerTail24_9, scan, if_neg (by decide), innerCandidates24_9_checked]
  change scan edge (search edge 3) 4 [103] 103 innerTail24_10
    (if 3 ≤ innerCandidates24_9.length then search edge 3 [103, 2467] innerCandidates24_9 2570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan24_10

def innerTail24_8 : List Nat := 2341 :: innerTail24_9
def innerCandidates24_8 : List Nat := [7489, 7879]
set_option maxRecDepth 4096 in
theorem innerCandidates24_8_checked : candidates edge 4 103 2341 innerTail24_9 best.bound = innerCandidates24_8 := by
  unfold candidates
  have row : edge 2341 = PrimePairCache.smallRow2341.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan24_8 : scan edge (search edge 3) 4 [103] 103 innerTail24_8 best = best := by
  rw [innerTail24_8, scan, if_neg (by decide), innerCandidates24_8_checked]
  change scan edge (search edge 3) 4 [103] 103 innerTail24_9
    (if 3 ≤ innerCandidates24_8.length then search edge 3 [103, 2341] innerCandidates24_8 2444 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan24_9

def innerTail24_7 : List Nat := 1999 :: innerTail24_8
def innerCandidates24_7 : List Nat := [5869]
set_option maxRecDepth 4096 in
theorem innerCandidates24_7_checked : candidates edge 4 103 1999 innerTail24_8 best.bound = innerCandidates24_7 := by
  unfold candidates
  have row : edge 1999 = PrimePairCache.smallRow1999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan24_7 : scan edge (search edge 3) 4 [103] 103 innerTail24_7 best = best := by
  rw [innerTail24_7, scan, if_neg (by decide), innerCandidates24_7_checked]
  change scan edge (search edge 3) 4 [103] 103 innerTail24_8
    (if 3 ≤ innerCandidates24_7.length then search edge 3 [103, 1999] innerCandidates24_7 2102 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan24_8

def innerTail24_6 : List Nat := 1831 :: innerTail24_7
def innerCandidates24_6 : List Nat := [2803]
set_option maxRecDepth 4096 in
theorem innerCandidates24_6_checked : candidates edge 4 103 1831 innerTail24_7 best.bound = innerCandidates24_6 := by
  unfold candidates
  have row : edge 1831 = PrimePairCache.smallRow1831.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan24_6 : scan edge (search edge 3) 4 [103] 103 innerTail24_6 best = best := by
  rw [innerTail24_6, scan, if_neg (by decide), innerCandidates24_6_checked]
  change scan edge (search edge 3) 4 [103] 103 innerTail24_7
    (if 3 ≤ innerCandidates24_6.length then search edge 3 [103, 1831] innerCandidates24_6 1934 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan24_7

def innerTail24_5 : List Nat := 1759 :: innerTail24_6
def innerCandidates24_5 : List Nat := [1831]
set_option maxRecDepth 4096 in
theorem innerCandidates24_5_checked : candidates edge 4 103 1759 innerTail24_6 best.bound = innerCandidates24_5 := by
  unfold candidates
  have row : edge 1759 = PrimePairCache.smallRow1759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan24_5 : scan edge (search edge 3) 4 [103] 103 innerTail24_5 best = best := by
  rw [innerTail24_5, scan, if_neg (by decide), innerCandidates24_5_checked]
  change scan edge (search edge 3) 4 [103] 103 innerTail24_6
    (if 3 ≤ innerCandidates24_5.length then search edge 3 [103, 1759] innerCandidates24_5 1862 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan24_6

def innerTail24_4 : List Nat := 1669 :: innerTail24_5
def innerCandidates24_4 : List Nat := [6661]
set_option maxRecDepth 4096 in
theorem innerCandidates24_4_checked : candidates edge 4 103 1669 innerTail24_5 best.bound = innerCandidates24_4 := by
  unfold candidates
  have row : edge 1669 = PrimePairCache.smallRow1669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan24_4 : scan edge (search edge 3) 4 [103] 103 innerTail24_4 best = best := by
  rw [innerTail24_4, scan, if_neg (by decide), innerCandidates24_4_checked]
  change scan edge (search edge 3) 4 [103] 103 innerTail24_5
    (if 3 ≤ innerCandidates24_4.length then search edge 3 [103, 1669] innerCandidates24_4 1772 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan24_5

def innerTail24_3 : List Nat := 1483 :: innerTail24_4
def innerCandidates24_3 : List Nat := [2341, 4357, 5869]
set_option maxRecDepth 4096 in
theorem innerCandidates24_3_checked : candidates edge 4 103 1483 innerTail24_4 best.bound = innerCandidates24_3 := by
  unfold candidates
  have row : edge 1483 = PrimePairCache.smallRow1483.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild24_3 : search edge 3 [103, 1483] innerCandidates24_3 1586 best = best := by decide

theorem innerScan24_3 : scan edge (search edge 3) 4 [103] 103 innerTail24_3 best = best := by
  rw [innerTail24_3, scan, if_neg (by decide), innerCandidates24_3_checked]
  change scan edge (search edge 3) 4 [103] 103 innerTail24_4
    (if 3 ≤ innerCandidates24_3.length then search edge 3 [103, 1483] innerCandidates24_3 1586 best else best) = best
  rw [if_pos (by decide), innerChild24_3]
  exact innerScan24_4

def innerTail24_2 : List Nat := 997 :: innerTail24_3
def innerCandidates24_2 : List Nat := [1483, 2617]
set_option maxRecDepth 4096 in
theorem innerCandidates24_2_checked : candidates edge 4 103 997 innerTail24_3 best.bound = innerCandidates24_2 := by
  unfold candidates
  have row : edge 997 = PrimePairCache.smallRow997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan24_2 : scan edge (search edge 3) 4 [103] 103 innerTail24_2 best = best := by
  rw [innerTail24_2, scan, if_neg (by decide), innerCandidates24_2_checked]
  change scan edge (search edge 3) 4 [103] 103 innerTail24_3
    (if 3 ≤ innerCandidates24_2.length then search edge 3 [103, 997] innerCandidates24_2 1100 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan24_3

def innerTail24_1 : List Nat := 421 :: innerTail24_2
def innerCandidates24_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates24_1_checked : candidates edge 4 103 421 innerTail24_2 best.bound = innerCandidates24_1 := by
  unfold candidates
  have row : edge 421 = PrimePairCache.smallRow421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan24_1 : scan edge (search edge 3) 4 [103] 103 innerTail24_1 best = best := by
  rw [innerTail24_1, scan, if_neg (by decide), innerCandidates24_1_checked]
  change scan edge (search edge 3) 4 [103] 103 innerTail24_2
    (if 3 ≤ innerCandidates24_1.length then search edge 3 [103, 421] innerCandidates24_1 524 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan24_2

def innerTail24_0 : List Nat := 307 :: innerTail24_1
def innerCandidates24_0 : List Nat := [2467]
set_option maxRecDepth 4096 in
theorem innerCandidates24_0_checked : candidates edge 4 103 307 innerTail24_1 best.bound = innerCandidates24_0 := by
  unfold candidates
  have row : edge 307 = PrimePairCache.smallRow307.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan24_0 : scan edge (search edge 3) 4 [103] 103 innerTail24_0 best = best := by
  rw [innerTail24_0, scan, if_neg (by decide), innerCandidates24_0_checked]
  change scan edge (search edge 3) 4 [103] 103 innerTail24_1
    (if 3 ≤ innerCandidates24_0.length then search edge 3 [103, 307] innerCandidates24_0 410 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan24_1

theorem rootBranch24 : search edge 4 [103] rootCandidates24 103 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [103] 103 innerTail24_0 best = best
  exact innerScan24_0

end ProjectEulerSolutions.PrimePairUpper.Search
