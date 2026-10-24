import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates317 : List Nat := [2423, 2729, 3089, 3947, 4001, 4271, 4517, 4937, 5051, 5783, 6203, 6269, 6491, 6791, 7121, 7193, 7247, 7559, 7937, 7949]
set_option maxRecDepth 4096 in
theorem rootCandidates317_checked : candidates edge 5 0 2129 tail318 best.bound = rootCandidates317 := by
  unfold candidates
  have row : edge 2129 = PrimePairCache.smallRow2129.contains := by funext q; rfl
  rw [row]
  decide

def innerTail317_20 : List Nat := []
theorem innerScan317_20 : scan edge (search edge 3) 4 [2129] 2129 innerTail317_20 best = best := rfl

def innerTail317_19 : List Nat := 7949 :: innerTail317_20
def innerCandidates317_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates317_19_checked : candidates edge 4 2129 7949 innerTail317_20 best.bound = innerCandidates317_19 := by
  unfold candidates
  have row : edge 7949 = SearchData.row7949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan317_19 : scan edge (search edge 3) 4 [2129] 2129 innerTail317_19 best = best := by
  rw [innerTail317_19, scan, if_neg (by decide), innerCandidates317_19_checked]
  change scan edge (search edge 3) 4 [2129] 2129 innerTail317_20
    (if 3 ≤ innerCandidates317_19.length then search edge 3 [2129, 7949] innerCandidates317_19 10078 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan317_20

def innerTail317_18 : List Nat := 7937 :: innerTail317_19
def innerCandidates317_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates317_18_checked : candidates edge 4 2129 7937 innerTail317_19 best.bound = innerCandidates317_18 := by
  unfold candidates
  have row : edge 7937 = SearchData.row7937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan317_18 : scan edge (search edge 3) 4 [2129] 2129 innerTail317_18 best = best := by
  rw [innerTail317_18, scan, if_neg (by decide), innerCandidates317_18_checked]
  change scan edge (search edge 3) 4 [2129] 2129 innerTail317_19
    (if 3 ≤ innerCandidates317_18.length then search edge 3 [2129, 7937] innerCandidates317_18 10066 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan317_19

def innerTail317_17 : List Nat := 7559 :: innerTail317_18
def innerCandidates317_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates317_17_checked : candidates edge 4 2129 7559 innerTail317_18 best.bound = innerCandidates317_17 := by
  unfold candidates
  have row : edge 7559 = SearchData.row7559.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan317_17 : scan edge (search edge 3) 4 [2129] 2129 innerTail317_17 best = best := by
  rw [innerTail317_17, scan, if_neg (by decide), innerCandidates317_17_checked]
  change scan edge (search edge 3) 4 [2129] 2129 innerTail317_18
    (if 3 ≤ innerCandidates317_17.length then search edge 3 [2129, 7559] innerCandidates317_17 9688 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan317_18

def innerTail317_16 : List Nat := 7247 :: innerTail317_17
def innerCandidates317_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates317_16_checked : candidates edge 4 2129 7247 innerTail317_17 best.bound = innerCandidates317_16 := by
  unfold candidates
  have row : edge 7247 = SearchData.row7247.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan317_16 : scan edge (search edge 3) 4 [2129] 2129 innerTail317_16 best = best := by
  rw [innerTail317_16, scan, if_neg (by decide), innerCandidates317_16_checked]
  change scan edge (search edge 3) 4 [2129] 2129 innerTail317_17
    (if 3 ≤ innerCandidates317_16.length then search edge 3 [2129, 7247] innerCandidates317_16 9376 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan317_17

def innerTail317_15 : List Nat := 7193 :: innerTail317_16
def innerCandidates317_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates317_15_checked : candidates edge 4 2129 7193 innerTail317_16 best.bound = innerCandidates317_15 := by
  unfold candidates
  have row : edge 7193 = SearchData.row7193.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan317_15 : scan edge (search edge 3) 4 [2129] 2129 innerTail317_15 best = best := by
  rw [innerTail317_15, scan, if_neg (by decide), innerCandidates317_15_checked]
  change scan edge (search edge 3) 4 [2129] 2129 innerTail317_16
    (if 3 ≤ innerCandidates317_15.length then search edge 3 [2129, 7193] innerCandidates317_15 9322 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan317_16

def innerTail317_14 : List Nat := 7121 :: innerTail317_15
def innerCandidates317_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates317_14_checked : candidates edge 4 2129 7121 innerTail317_15 best.bound = innerCandidates317_14 := by
  unfold candidates
  have row : edge 7121 = SearchData.row7121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan317_14 : scan edge (search edge 3) 4 [2129] 2129 innerTail317_14 best = best := by
  rw [innerTail317_14, scan, if_neg (by decide), innerCandidates317_14_checked]
  change scan edge (search edge 3) 4 [2129] 2129 innerTail317_15
    (if 3 ≤ innerCandidates317_14.length then search edge 3 [2129, 7121] innerCandidates317_14 9250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan317_15

def innerTail317_13 : List Nat := 6791 :: innerTail317_14
def innerCandidates317_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates317_13_checked : candidates edge 4 2129 6791 innerTail317_14 best.bound = innerCandidates317_13 := by
  unfold candidates
  have row : edge 6791 = SearchData.row6791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan317_13 : scan edge (search edge 3) 4 [2129] 2129 innerTail317_13 best = best := by
  rw [innerTail317_13, scan, if_neg (by decide), innerCandidates317_13_checked]
  change scan edge (search edge 3) 4 [2129] 2129 innerTail317_14
    (if 3 ≤ innerCandidates317_13.length then search edge 3 [2129, 6791] innerCandidates317_13 8920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan317_14

def innerTail317_12 : List Nat := 6491 :: innerTail317_13
def innerCandidates317_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates317_12_checked : candidates edge 4 2129 6491 innerTail317_13 best.bound = innerCandidates317_12 := by
  unfold candidates
  have row : edge 6491 = SearchData.row6491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan317_12 : scan edge (search edge 3) 4 [2129] 2129 innerTail317_12 best = best := by
  rw [innerTail317_12, scan, if_neg (by decide), innerCandidates317_12_checked]
  change scan edge (search edge 3) 4 [2129] 2129 innerTail317_13
    (if 3 ≤ innerCandidates317_12.length then search edge 3 [2129, 6491] innerCandidates317_12 8620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan317_13

def innerTail317_11 : List Nat := 6269 :: innerTail317_12
def innerCandidates317_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates317_11_checked : candidates edge 4 2129 6269 innerTail317_12 best.bound = innerCandidates317_11 := by
  unfold candidates
  have row : edge 6269 = SearchData.row6269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan317_11 : scan edge (search edge 3) 4 [2129] 2129 innerTail317_11 best = best := by
  rw [innerTail317_11, scan, if_neg (by decide), innerCandidates317_11_checked]
  change scan edge (search edge 3) 4 [2129] 2129 innerTail317_12
    (if 3 ≤ innerCandidates317_11.length then search edge 3 [2129, 6269] innerCandidates317_11 8398 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan317_12

def innerTail317_10 : List Nat := 6203 :: innerTail317_11
def innerCandidates317_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates317_10_checked : candidates edge 4 2129 6203 innerTail317_11 best.bound = innerCandidates317_10 := by
  unfold candidates
  have row : edge 6203 = SearchData.row6203.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan317_10 : scan edge (search edge 3) 4 [2129] 2129 innerTail317_10 best = best := by
  rw [innerTail317_10, scan, if_neg (by decide), innerCandidates317_10_checked]
  change scan edge (search edge 3) 4 [2129] 2129 innerTail317_11
    (if 3 ≤ innerCandidates317_10.length then search edge 3 [2129, 6203] innerCandidates317_10 8332 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan317_11

def innerTail317_9 : List Nat := 5783 :: innerTail317_10
def innerCandidates317_9 : List Nat := [6491]
set_option maxRecDepth 4096 in
theorem innerCandidates317_9_checked : candidates edge 4 2129 5783 innerTail317_10 best.bound = innerCandidates317_9 := by
  unfold candidates
  have row : edge 5783 = SearchData.row5783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan317_9 : scan edge (search edge 3) 4 [2129] 2129 innerTail317_9 best = best := by
  rw [innerTail317_9, scan, if_neg (by decide), innerCandidates317_9_checked]
  change scan edge (search edge 3) 4 [2129] 2129 innerTail317_10
    (if 3 ≤ innerCandidates317_9.length then search edge 3 [2129, 5783] innerCandidates317_9 7912 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan317_10

def innerTail317_8 : List Nat := 5051 :: innerTail317_9
def innerCandidates317_8 : List Nat := [7949]
set_option maxRecDepth 4096 in
theorem innerCandidates317_8_checked : candidates edge 4 2129 5051 innerTail317_9 best.bound = innerCandidates317_8 := by
  unfold candidates
  have row : edge 5051 = SearchData.row5051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan317_8 : scan edge (search edge 3) 4 [2129] 2129 innerTail317_8 best = best := by
  rw [innerTail317_8, scan, if_neg (by decide), innerCandidates317_8_checked]
  change scan edge (search edge 3) 4 [2129] 2129 innerTail317_9
    (if 3 ≤ innerCandidates317_8.length then search edge 3 [2129, 5051] innerCandidates317_8 7180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan317_9

def innerTail317_7 : List Nat := 4937 :: innerTail317_8
def innerCandidates317_7 : List Nat := [6491, 7247]
set_option maxRecDepth 4096 in
theorem innerCandidates317_7_checked : candidates edge 4 2129 4937 innerTail317_8 best.bound = innerCandidates317_7 := by
  unfold candidates
  have row : edge 4937 = SearchData.row4937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan317_7 : scan edge (search edge 3) 4 [2129] 2129 innerTail317_7 best = best := by
  rw [innerTail317_7, scan, if_neg (by decide), innerCandidates317_7_checked]
  change scan edge (search edge 3) 4 [2129] 2129 innerTail317_8
    (if 3 ≤ innerCandidates317_7.length then search edge 3 [2129, 4937] innerCandidates317_7 7066 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan317_8

def innerTail317_6 : List Nat := 4517 :: innerTail317_7
def innerCandidates317_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates317_6_checked : candidates edge 4 2129 4517 innerTail317_7 best.bound = innerCandidates317_6 := by
  unfold candidates
  have row : edge 4517 = SearchData.row4517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan317_6 : scan edge (search edge 3) 4 [2129] 2129 innerTail317_6 best = best := by
  rw [innerTail317_6, scan, if_neg (by decide), innerCandidates317_6_checked]
  change scan edge (search edge 3) 4 [2129] 2129 innerTail317_7
    (if 3 ≤ innerCandidates317_6.length then search edge 3 [2129, 4517] innerCandidates317_6 6646 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan317_7

def innerTail317_5 : List Nat := 4271 :: innerTail317_6
def innerCandidates317_5 : List Nat := [4517]
set_option maxRecDepth 4096 in
theorem innerCandidates317_5_checked : candidates edge 4 2129 4271 innerTail317_6 best.bound = innerCandidates317_5 := by
  unfold candidates
  have row : edge 4271 = SearchData.row4271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan317_5 : scan edge (search edge 3) 4 [2129] 2129 innerTail317_5 best = best := by
  rw [innerTail317_5, scan, if_neg (by decide), innerCandidates317_5_checked]
  change scan edge (search edge 3) 4 [2129] 2129 innerTail317_6
    (if 3 ≤ innerCandidates317_5.length then search edge 3 [2129, 4271] innerCandidates317_5 6400 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan317_6

def innerTail317_4 : List Nat := 4001 :: innerTail317_5
def innerCandidates317_4 : List Nat := [5783, 7121]
set_option maxRecDepth 4096 in
theorem innerCandidates317_4_checked : candidates edge 4 2129 4001 innerTail317_5 best.bound = innerCandidates317_4 := by
  unfold candidates
  have row : edge 4001 = SearchData.row4001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan317_4 : scan edge (search edge 3) 4 [2129] 2129 innerTail317_4 best = best := by
  rw [innerTail317_4, scan, if_neg (by decide), innerCandidates317_4_checked]
  change scan edge (search edge 3) 4 [2129] 2129 innerTail317_5
    (if 3 ≤ innerCandidates317_4.length then search edge 3 [2129, 4001] innerCandidates317_4 6130 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan317_5

def innerTail317_3 : List Nat := 3947 :: innerTail317_4
def innerCandidates317_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates317_3_checked : candidates edge 4 2129 3947 innerTail317_4 best.bound = innerCandidates317_3 := by
  unfold candidates
  have row : edge 3947 = SearchData.row3947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan317_3 : scan edge (search edge 3) 4 [2129] 2129 innerTail317_3 best = best := by
  rw [innerTail317_3, scan, if_neg (by decide), innerCandidates317_3_checked]
  change scan edge (search edge 3) 4 [2129] 2129 innerTail317_4
    (if 3 ≤ innerCandidates317_3.length then search edge 3 [2129, 3947] innerCandidates317_3 6076 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan317_4

def innerTail317_2 : List Nat := 3089 :: innerTail317_3
def innerCandidates317_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates317_2_checked : candidates edge 4 2129 3089 innerTail317_3 best.bound = innerCandidates317_2 := by
  unfold candidates
  have row : edge 3089 = SearchData.row3089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan317_2 : scan edge (search edge 3) 4 [2129] 2129 innerTail317_2 best = best := by
  rw [innerTail317_2, scan, if_neg (by decide), innerCandidates317_2_checked]
  change scan edge (search edge 3) 4 [2129] 2129 innerTail317_3
    (if 3 ≤ innerCandidates317_2.length then search edge 3 [2129, 3089] innerCandidates317_2 5218 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan317_3

def innerTail317_1 : List Nat := 2729 :: innerTail317_2
def innerCandidates317_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates317_1_checked : candidates edge 4 2129 2729 innerTail317_2 best.bound = innerCandidates317_1 := by
  unfold candidates
  have row : edge 2729 = SearchData.row2729.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan317_1 : scan edge (search edge 3) 4 [2129] 2129 innerTail317_1 best = best := by
  rw [innerTail317_1, scan, if_neg (by decide), innerCandidates317_1_checked]
  change scan edge (search edge 3) 4 [2129] 2129 innerTail317_2
    (if 3 ≤ innerCandidates317_1.length then search edge 3 [2129, 2729] innerCandidates317_1 4858 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan317_2

def innerTail317_0 : List Nat := 2423 :: innerTail317_1
def innerCandidates317_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates317_0_checked : candidates edge 4 2129 2423 innerTail317_1 best.bound = innerCandidates317_0 := by
  unfold candidates
  have row : edge 2423 = PrimePairCache.smallRow2423.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan317_0 : scan edge (search edge 3) 4 [2129] 2129 innerTail317_0 best = best := by
  rw [innerTail317_0, scan, if_neg (by decide), innerCandidates317_0_checked]
  change scan edge (search edge 3) 4 [2129] 2129 innerTail317_1
    (if 3 ≤ innerCandidates317_0.length then search edge 3 [2129, 2423] innerCandidates317_0 4552 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan317_1

theorem rootBranch317 : search edge 4 [2129] rootCandidates317 2129 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2129] 2129 innerTail317_0 best = best
  exact innerScan317_0

def rootCandidates316 : List Nat := [2131, 2659, 3121, 3433, 3583, 3727, 4093, 4507, 5227, 5419, 5431, 6067, 6277, 6481, 6553, 7309, 7393, 7927]
set_option maxRecDepth 4096 in
theorem rootCandidates316_checked : candidates edge 5 0 2113 tail317 best.bound = rootCandidates316 := by
  unfold candidates
  have row : edge 2113 = PrimePairCache.smallRow2113.contains := by funext q; rfl
  rw [row]
  decide

def innerTail316_18 : List Nat := []
theorem innerScan316_18 : scan edge (search edge 3) 4 [2113] 2113 innerTail316_18 best = best := rfl

def innerTail316_17 : List Nat := 7927 :: innerTail316_18
def innerCandidates316_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates316_17_checked : candidates edge 4 2113 7927 innerTail316_18 best.bound = innerCandidates316_17 := by
  unfold candidates
  have row : edge 7927 = SearchData.row7927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan316_17 : scan edge (search edge 3) 4 [2113] 2113 innerTail316_17 best = best := by
  rw [innerTail316_17, scan, if_neg (by decide), innerCandidates316_17_checked]
  change scan edge (search edge 3) 4 [2113] 2113 innerTail316_18
    (if 3 ≤ innerCandidates316_17.length then search edge 3 [2113, 7927] innerCandidates316_17 10040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan316_18

def innerTail316_16 : List Nat := 7393 :: innerTail316_17
def innerCandidates316_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates316_16_checked : candidates edge 4 2113 7393 innerTail316_17 best.bound = innerCandidates316_16 := by
  unfold candidates
  have row : edge 7393 = SearchData.row7393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan316_16 : scan edge (search edge 3) 4 [2113] 2113 innerTail316_16 best = best := by
  rw [innerTail316_16, scan, if_neg (by decide), innerCandidates316_16_checked]
  change scan edge (search edge 3) 4 [2113] 2113 innerTail316_17
    (if 3 ≤ innerCandidates316_16.length then search edge 3 [2113, 7393] innerCandidates316_16 9506 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan316_17

def innerTail316_15 : List Nat := 7309 :: innerTail316_16
def innerCandidates316_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates316_15_checked : candidates edge 4 2113 7309 innerTail316_16 best.bound = innerCandidates316_15 := by
  unfold candidates
  have row : edge 7309 = SearchData.row7309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan316_15 : scan edge (search edge 3) 4 [2113] 2113 innerTail316_15 best = best := by
  rw [innerTail316_15, scan, if_neg (by decide), innerCandidates316_15_checked]
  change scan edge (search edge 3) 4 [2113] 2113 innerTail316_16
    (if 3 ≤ innerCandidates316_15.length then search edge 3 [2113, 7309] innerCandidates316_15 9422 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan316_16

def innerTail316_14 : List Nat := 6553 :: innerTail316_15
def innerCandidates316_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates316_14_checked : candidates edge 4 2113 6553 innerTail316_15 best.bound = innerCandidates316_14 := by
  unfold candidates
  have row : edge 6553 = SearchData.row6553.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan316_14 : scan edge (search edge 3) 4 [2113] 2113 innerTail316_14 best = best := by
  rw [innerTail316_14, scan, if_neg (by decide), innerCandidates316_14_checked]
  change scan edge (search edge 3) 4 [2113] 2113 innerTail316_15
    (if 3 ≤ innerCandidates316_14.length then search edge 3 [2113, 6553] innerCandidates316_14 8666 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan316_15

def innerTail316_13 : List Nat := 6481 :: innerTail316_14
def innerCandidates316_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates316_13_checked : candidates edge 4 2113 6481 innerTail316_14 best.bound = innerCandidates316_13 := by
  unfold candidates
  have row : edge 6481 = SearchData.row6481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan316_13 : scan edge (search edge 3) 4 [2113] 2113 innerTail316_13 best = best := by
  rw [innerTail316_13, scan, if_neg (by decide), innerCandidates316_13_checked]
  change scan edge (search edge 3) 4 [2113] 2113 innerTail316_14
    (if 3 ≤ innerCandidates316_13.length then search edge 3 [2113, 6481] innerCandidates316_13 8594 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan316_14

def innerTail316_12 : List Nat := 6277 :: innerTail316_13
def innerCandidates316_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates316_12_checked : candidates edge 4 2113 6277 innerTail316_13 best.bound = innerCandidates316_12 := by
  unfold candidates
  have row : edge 6277 = SearchData.row6277.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan316_12 : scan edge (search edge 3) 4 [2113] 2113 innerTail316_12 best = best := by
  rw [innerTail316_12, scan, if_neg (by decide), innerCandidates316_12_checked]
  change scan edge (search edge 3) 4 [2113] 2113 innerTail316_13
    (if 3 ≤ innerCandidates316_12.length then search edge 3 [2113, 6277] innerCandidates316_12 8390 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan316_13

def innerTail316_11 : List Nat := 6067 :: innerTail316_12
def innerCandidates316_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates316_11_checked : candidates edge 4 2113 6067 innerTail316_12 best.bound = innerCandidates316_11 := by
  unfold candidates
  have row : edge 6067 = SearchData.row6067.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan316_11 : scan edge (search edge 3) 4 [2113] 2113 innerTail316_11 best = best := by
  rw [innerTail316_11, scan, if_neg (by decide), innerCandidates316_11_checked]
  change scan edge (search edge 3) 4 [2113] 2113 innerTail316_12
    (if 3 ≤ innerCandidates316_11.length then search edge 3 [2113, 6067] innerCandidates316_11 8180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan316_12

def innerTail316_10 : List Nat := 5431 :: innerTail316_11
def innerCandidates316_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates316_10_checked : candidates edge 4 2113 5431 innerTail316_11 best.bound = innerCandidates316_10 := by
  unfold candidates
  have row : edge 5431 = SearchData.row5431.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan316_10 : scan edge (search edge 3) 4 [2113] 2113 innerTail316_10 best = best := by
  rw [innerTail316_10, scan, if_neg (by decide), innerCandidates316_10_checked]
  change scan edge (search edge 3) 4 [2113] 2113 innerTail316_11
    (if 3 ≤ innerCandidates316_10.length then search edge 3 [2113, 5431] innerCandidates316_10 7544 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan316_11

def innerTail316_9 : List Nat := 5419 :: innerTail316_10
def innerCandidates316_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates316_9_checked : candidates edge 4 2113 5419 innerTail316_10 best.bound = innerCandidates316_9 := by
  unfold candidates
  have row : edge 5419 = SearchData.row5419.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan316_9 : scan edge (search edge 3) 4 [2113] 2113 innerTail316_9 best = best := by
  rw [innerTail316_9, scan, if_neg (by decide), innerCandidates316_9_checked]
  change scan edge (search edge 3) 4 [2113] 2113 innerTail316_10
    (if 3 ≤ innerCandidates316_9.length then search edge 3 [2113, 5419] innerCandidates316_9 7532 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan316_10

def innerTail316_8 : List Nat := 5227 :: innerTail316_9
def innerCandidates316_8 : List Nat := [6481]
set_option maxRecDepth 4096 in
theorem innerCandidates316_8_checked : candidates edge 4 2113 5227 innerTail316_9 best.bound = innerCandidates316_8 := by
  unfold candidates
  have row : edge 5227 = SearchData.row5227.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan316_8 : scan edge (search edge 3) 4 [2113] 2113 innerTail316_8 best = best := by
  rw [innerTail316_8, scan, if_neg (by decide), innerCandidates316_8_checked]
  change scan edge (search edge 3) 4 [2113] 2113 innerTail316_9
    (if 3 ≤ innerCandidates316_8.length then search edge 3 [2113, 5227] innerCandidates316_8 7340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan316_9

def innerTail316_7 : List Nat := 4507 :: innerTail316_8
def innerCandidates316_7 : List Nat := [5419]
set_option maxRecDepth 4096 in
theorem innerCandidates316_7_checked : candidates edge 4 2113 4507 innerTail316_8 best.bound = innerCandidates316_7 := by
  unfold candidates
  have row : edge 4507 = SearchData.row4507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan316_7 : scan edge (search edge 3) 4 [2113] 2113 innerTail316_7 best = best := by
  rw [innerTail316_7, scan, if_neg (by decide), innerCandidates316_7_checked]
  change scan edge (search edge 3) 4 [2113] 2113 innerTail316_8
    (if 3 ≤ innerCandidates316_7.length then search edge 3 [2113, 4507] innerCandidates316_7 6620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan316_8

def innerTail316_6 : List Nat := 4093 :: innerTail316_7
def innerCandidates316_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates316_6_checked : candidates edge 4 2113 4093 innerTail316_7 best.bound = innerCandidates316_6 := by
  unfold candidates
  have row : edge 4093 = SearchData.row4093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan316_6 : scan edge (search edge 3) 4 [2113] 2113 innerTail316_6 best = best := by
  rw [innerTail316_6, scan, if_neg (by decide), innerCandidates316_6_checked]
  change scan edge (search edge 3) 4 [2113] 2113 innerTail316_7
    (if 3 ≤ innerCandidates316_6.length then search edge 3 [2113, 4093] innerCandidates316_6 6206 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan316_7

def innerTail316_5 : List Nat := 3727 :: innerTail316_6
def innerCandidates316_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates316_5_checked : candidates edge 4 2113 3727 innerTail316_6 best.bound = innerCandidates316_5 := by
  unfold candidates
  have row : edge 3727 = SearchData.row3727.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan316_5 : scan edge (search edge 3) 4 [2113] 2113 innerTail316_5 best = best := by
  rw [innerTail316_5, scan, if_neg (by decide), innerCandidates316_5_checked]
  change scan edge (search edge 3) 4 [2113] 2113 innerTail316_6
    (if 3 ≤ innerCandidates316_5.length then search edge 3 [2113, 3727] innerCandidates316_5 5840 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan316_6

def innerTail316_4 : List Nat := 3583 :: innerTail316_5
def innerCandidates316_4 : List Nat := [6067]
set_option maxRecDepth 4096 in
theorem innerCandidates316_4_checked : candidates edge 4 2113 3583 innerTail316_5 best.bound = innerCandidates316_4 := by
  unfold candidates
  have row : edge 3583 = SearchData.row3583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan316_4 : scan edge (search edge 3) 4 [2113] 2113 innerTail316_4 best = best := by
  rw [innerTail316_4, scan, if_neg (by decide), innerCandidates316_4_checked]
  change scan edge (search edge 3) 4 [2113] 2113 innerTail316_5
    (if 3 ≤ innerCandidates316_4.length then search edge 3 [2113, 3583] innerCandidates316_4 5696 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan316_5

def innerTail316_3 : List Nat := 3433 :: innerTail316_4
def innerCandidates316_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates316_3_checked : candidates edge 4 2113 3433 innerTail316_4 best.bound = innerCandidates316_3 := by
  unfold candidates
  have row : edge 3433 = SearchData.row3433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan316_3 : scan edge (search edge 3) 4 [2113] 2113 innerTail316_3 best = best := by
  rw [innerTail316_3, scan, if_neg (by decide), innerCandidates316_3_checked]
  change scan edge (search edge 3) 4 [2113] 2113 innerTail316_4
    (if 3 ≤ innerCandidates316_3.length then search edge 3 [2113, 3433] innerCandidates316_3 5546 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan316_4

def innerTail316_2 : List Nat := 3121 :: innerTail316_3
def innerCandidates316_2 : List Nat := [3433, 4507, 7393]
set_option maxRecDepth 4096 in
theorem innerCandidates316_2_checked : candidates edge 4 2113 3121 innerTail316_3 best.bound = innerCandidates316_2 := by
  unfold candidates
  have row : edge 3121 = SearchData.row3121.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild316_2 : search edge 3 [2113, 3121] innerCandidates316_2 5234 best = best := by decide

theorem innerScan316_2 : scan edge (search edge 3) 4 [2113] 2113 innerTail316_2 best = best := by
  rw [innerTail316_2, scan, if_neg (by decide), innerCandidates316_2_checked]
  change scan edge (search edge 3) 4 [2113] 2113 innerTail316_3
    (if 3 ≤ innerCandidates316_2.length then search edge 3 [2113, 3121] innerCandidates316_2 5234 best else best) = best
  rw [if_pos (by decide), innerChild316_2]
  exact innerScan316_3

def innerTail316_1 : List Nat := 2659 :: innerTail316_2
def innerCandidates316_1 : List Nat := [7927]
set_option maxRecDepth 4096 in
theorem innerCandidates316_1_checked : candidates edge 4 2113 2659 innerTail316_2 best.bound = innerCandidates316_1 := by
  unfold candidates
  have row : edge 2659 = PrimePairCache.smallRow2659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan316_1 : scan edge (search edge 3) 4 [2113] 2113 innerTail316_1 best = best := by
  rw [innerTail316_1, scan, if_neg (by decide), innerCandidates316_1_checked]
  change scan edge (search edge 3) 4 [2113] 2113 innerTail316_2
    (if 3 ≤ innerCandidates316_1.length then search edge 3 [2113, 2659] innerCandidates316_1 4772 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan316_2

def innerTail316_0 : List Nat := 2131 :: innerTail316_1
def innerCandidates316_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates316_0_checked : candidates edge 4 2113 2131 innerTail316_1 best.bound = innerCandidates316_0 := by
  unfold candidates
  have row : edge 2131 = PrimePairCache.smallRow2131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan316_0 : scan edge (search edge 3) 4 [2113] 2113 innerTail316_0 best = best := by
  rw [innerTail316_0, scan, if_neg (by decide), innerCandidates316_0_checked]
  change scan edge (search edge 3) 4 [2113] 2113 innerTail316_1
    (if 3 ≤ innerCandidates316_0.length then search edge 3 [2113, 2131] innerCandidates316_0 4244 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan316_1

theorem rootBranch316 : search edge 4 [2113] rootCandidates316 2113 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2113] 2113 innerTail316_0 best = best
  exact innerScan316_0

def rootCandidates315 : List Nat := [2411, 2969, 2999, 3041, 3083, 3221, 3251, 3329, 3467, 4019, 4127, 4217, 4229, 4649, 4931, 5261, 6857, 6911, 7487, 7793, 8219]
set_option maxRecDepth 4096 in
theorem rootCandidates315_checked : candidates edge 5 0 2111 tail316 best.bound = rootCandidates315 := by
  unfold candidates
  have row : edge 2111 = PrimePairCache.smallRow2111.contains := by funext q; rfl
  rw [row]
  decide

def innerTail315_21 : List Nat := []
theorem innerScan315_21 : scan edge (search edge 3) 4 [2111] 2111 innerTail315_21 best = best := rfl

def innerTail315_20 : List Nat := 8219 :: innerTail315_21
def innerCandidates315_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates315_20_checked : candidates edge 4 2111 8219 innerTail315_21 best.bound = innerCandidates315_20 := by
  unfold candidates
  have row : edge 8219 = SearchData.row8219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan315_20 : scan edge (search edge 3) 4 [2111] 2111 innerTail315_20 best = best := by
  rw [innerTail315_20, scan, if_neg (by decide), innerCandidates315_20_checked]
  change scan edge (search edge 3) 4 [2111] 2111 innerTail315_21
    (if 3 ≤ innerCandidates315_20.length then search edge 3 [2111, 8219] innerCandidates315_20 10330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan315_21

def innerTail315_19 : List Nat := 7793 :: innerTail315_20
def innerCandidates315_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates315_19_checked : candidates edge 4 2111 7793 innerTail315_20 best.bound = innerCandidates315_19 := by
  unfold candidates
  have row : edge 7793 = SearchData.row7793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan315_19 : scan edge (search edge 3) 4 [2111] 2111 innerTail315_19 best = best := by
  rw [innerTail315_19, scan, if_neg (by decide), innerCandidates315_19_checked]
  change scan edge (search edge 3) 4 [2111] 2111 innerTail315_20
    (if 3 ≤ innerCandidates315_19.length then search edge 3 [2111, 7793] innerCandidates315_19 9904 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan315_20

def innerTail315_18 : List Nat := 7487 :: innerTail315_19
def innerCandidates315_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates315_18_checked : candidates edge 4 2111 7487 innerTail315_19 best.bound = innerCandidates315_18 := by
  unfold candidates
  have row : edge 7487 = SearchData.row7487.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan315_18 : scan edge (search edge 3) 4 [2111] 2111 innerTail315_18 best = best := by
  rw [innerTail315_18, scan, if_neg (by decide), innerCandidates315_18_checked]
  change scan edge (search edge 3) 4 [2111] 2111 innerTail315_19
    (if 3 ≤ innerCandidates315_18.length then search edge 3 [2111, 7487] innerCandidates315_18 9598 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan315_19

def innerTail315_17 : List Nat := 6911 :: innerTail315_18
def innerCandidates315_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates315_17_checked : candidates edge 4 2111 6911 innerTail315_18 best.bound = innerCandidates315_17 := by
  unfold candidates
  have row : edge 6911 = SearchData.row6911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan315_17 : scan edge (search edge 3) 4 [2111] 2111 innerTail315_17 best = best := by
  rw [innerTail315_17, scan, if_neg (by decide), innerCandidates315_17_checked]
  change scan edge (search edge 3) 4 [2111] 2111 innerTail315_18
    (if 3 ≤ innerCandidates315_17.length then search edge 3 [2111, 6911] innerCandidates315_17 9022 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan315_18

def innerTail315_16 : List Nat := 6857 :: innerTail315_17
def innerCandidates315_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates315_16_checked : candidates edge 4 2111 6857 innerTail315_17 best.bound = innerCandidates315_16 := by
  unfold candidates
  have row : edge 6857 = SearchData.row6857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan315_16 : scan edge (search edge 3) 4 [2111] 2111 innerTail315_16 best = best := by
  rw [innerTail315_16, scan, if_neg (by decide), innerCandidates315_16_checked]
  change scan edge (search edge 3) 4 [2111] 2111 innerTail315_17
    (if 3 ≤ innerCandidates315_16.length then search edge 3 [2111, 6857] innerCandidates315_16 8968 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan315_17

def innerTail315_15 : List Nat := 5261 :: innerTail315_16
def innerCandidates315_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates315_15_checked : candidates edge 4 2111 5261 innerTail315_16 best.bound = innerCandidates315_15 := by
  unfold candidates
  have row : edge 5261 = SearchData.row5261.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan315_15 : scan edge (search edge 3) 4 [2111] 2111 innerTail315_15 best = best := by
  rw [innerTail315_15, scan, if_neg (by decide), innerCandidates315_15_checked]
  change scan edge (search edge 3) 4 [2111] 2111 innerTail315_16
    (if 3 ≤ innerCandidates315_15.length then search edge 3 [2111, 5261] innerCandidates315_15 7372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan315_16

def innerTail315_14 : List Nat := 4931 :: innerTail315_15
def innerCandidates315_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates315_14_checked : candidates edge 4 2111 4931 innerTail315_15 best.bound = innerCandidates315_14 := by
  unfold candidates
  have row : edge 4931 = SearchData.row4931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan315_14 : scan edge (search edge 3) 4 [2111] 2111 innerTail315_14 best = best := by
  rw [innerTail315_14, scan, if_neg (by decide), innerCandidates315_14_checked]
  change scan edge (search edge 3) 4 [2111] 2111 innerTail315_15
    (if 3 ≤ innerCandidates315_14.length then search edge 3 [2111, 4931] innerCandidates315_14 7042 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan315_15

def innerTail315_13 : List Nat := 4649 :: innerTail315_14
def innerCandidates315_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates315_13_checked : candidates edge 4 2111 4649 innerTail315_14 best.bound = innerCandidates315_13 := by
  unfold candidates
  have row : edge 4649 = SearchData.row4649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan315_13 : scan edge (search edge 3) 4 [2111] 2111 innerTail315_13 best = best := by
  rw [innerTail315_13, scan, if_neg (by decide), innerCandidates315_13_checked]
  change scan edge (search edge 3) 4 [2111] 2111 innerTail315_14
    (if 3 ≤ innerCandidates315_13.length then search edge 3 [2111, 4649] innerCandidates315_13 6760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan315_14

def innerTail315_12 : List Nat := 4229 :: innerTail315_13
def innerCandidates315_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates315_12_checked : candidates edge 4 2111 4229 innerTail315_13 best.bound = innerCandidates315_12 := by
  unfold candidates
  have row : edge 4229 = SearchData.row4229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan315_12 : scan edge (search edge 3) 4 [2111] 2111 innerTail315_12 best = best := by
  rw [innerTail315_12, scan, if_neg (by decide), innerCandidates315_12_checked]
  change scan edge (search edge 3) 4 [2111] 2111 innerTail315_13
    (if 3 ≤ innerCandidates315_12.length then search edge 3 [2111, 4229] innerCandidates315_12 6340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan315_13

def innerTail315_11 : List Nat := 4217 :: innerTail315_12
def innerCandidates315_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates315_11_checked : candidates edge 4 2111 4217 innerTail315_12 best.bound = innerCandidates315_11 := by
  unfold candidates
  have row : edge 4217 = SearchData.row4217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan315_11 : scan edge (search edge 3) 4 [2111] 2111 innerTail315_11 best = best := by
  rw [innerTail315_11, scan, if_neg (by decide), innerCandidates315_11_checked]
  change scan edge (search edge 3) 4 [2111] 2111 innerTail315_12
    (if 3 ≤ innerCandidates315_11.length then search edge 3 [2111, 4217] innerCandidates315_11 6328 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan315_12

def innerTail315_10 : List Nat := 4127 :: innerTail315_11
def innerCandidates315_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates315_10_checked : candidates edge 4 2111 4127 innerTail315_11 best.bound = innerCandidates315_10 := by
  unfold candidates
  have row : edge 4127 = SearchData.row4127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan315_10 : scan edge (search edge 3) 4 [2111] 2111 innerTail315_10 best = best := by
  rw [innerTail315_10, scan, if_neg (by decide), innerCandidates315_10_checked]
  change scan edge (search edge 3) 4 [2111] 2111 innerTail315_11
    (if 3 ≤ innerCandidates315_10.length then search edge 3 [2111, 4127] innerCandidates315_10 6238 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan315_11

def innerTail315_9 : List Nat := 4019 :: innerTail315_10
def innerCandidates315_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates315_9_checked : candidates edge 4 2111 4019 innerTail315_10 best.bound = innerCandidates315_9 := by
  unfold candidates
  have row : edge 4019 = SearchData.row4019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan315_9 : scan edge (search edge 3) 4 [2111] 2111 innerTail315_9 best = best := by
  rw [innerTail315_9, scan, if_neg (by decide), innerCandidates315_9_checked]
  change scan edge (search edge 3) 4 [2111] 2111 innerTail315_10
    (if 3 ≤ innerCandidates315_9.length then search edge 3 [2111, 4019] innerCandidates315_9 6130 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan315_10

def innerTail315_8 : List Nat := 3467 :: innerTail315_9
def innerCandidates315_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates315_8_checked : candidates edge 4 2111 3467 innerTail315_9 best.bound = innerCandidates315_8 := by
  unfold candidates
  have row : edge 3467 = SearchData.row3467.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan315_8 : scan edge (search edge 3) 4 [2111] 2111 innerTail315_8 best = best := by
  rw [innerTail315_8, scan, if_neg (by decide), innerCandidates315_8_checked]
  change scan edge (search edge 3) 4 [2111] 2111 innerTail315_9
    (if 3 ≤ innerCandidates315_8.length then search edge 3 [2111, 3467] innerCandidates315_8 5578 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan315_9

def innerTail315_7 : List Nat := 3329 :: innerTail315_8
def innerCandidates315_7 : List Nat := [4127, 4649]
set_option maxRecDepth 4096 in
theorem innerCandidates315_7_checked : candidates edge 4 2111 3329 innerTail315_8 best.bound = innerCandidates315_7 := by
  unfold candidates
  have row : edge 3329 = SearchData.row3329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan315_7 : scan edge (search edge 3) 4 [2111] 2111 innerTail315_7 best = best := by
  rw [innerTail315_7, scan, if_neg (by decide), innerCandidates315_7_checked]
  change scan edge (search edge 3) 4 [2111] 2111 innerTail315_8
    (if 3 ≤ innerCandidates315_7.length then search edge 3 [2111, 3329] innerCandidates315_7 5440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan315_8

def innerTail315_6 : List Nat := 3251 :: innerTail315_7
def innerCandidates315_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates315_6_checked : candidates edge 4 2111 3251 innerTail315_7 best.bound = innerCandidates315_6 := by
  unfold candidates
  have row : edge 3251 = SearchData.row3251.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan315_6 : scan edge (search edge 3) 4 [2111] 2111 innerTail315_6 best = best := by
  rw [innerTail315_6, scan, if_neg (by decide), innerCandidates315_6_checked]
  change scan edge (search edge 3) 4 [2111] 2111 innerTail315_7
    (if 3 ≤ innerCandidates315_6.length then search edge 3 [2111, 3251] innerCandidates315_6 5362 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan315_7

def innerTail315_5 : List Nat := 3221 :: innerTail315_6
def innerCandidates315_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates315_5_checked : candidates edge 4 2111 3221 innerTail315_6 best.bound = innerCandidates315_5 := by
  unfold candidates
  have row : edge 3221 = SearchData.row3221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan315_5 : scan edge (search edge 3) 4 [2111] 2111 innerTail315_5 best = best := by
  rw [innerTail315_5, scan, if_neg (by decide), innerCandidates315_5_checked]
  change scan edge (search edge 3) 4 [2111] 2111 innerTail315_6
    (if 3 ≤ innerCandidates315_5.length then search edge 3 [2111, 3221] innerCandidates315_5 5332 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan315_6

def innerTail315_4 : List Nat := 3083 :: innerTail315_5
def innerCandidates315_4 : List Nat := [3329, 4931]
set_option maxRecDepth 4096 in
theorem innerCandidates315_4_checked : candidates edge 4 2111 3083 innerTail315_5 best.bound = innerCandidates315_4 := by
  unfold candidates
  have row : edge 3083 = SearchData.row3083.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan315_4 : scan edge (search edge 3) 4 [2111] 2111 innerTail315_4 best = best := by
  rw [innerTail315_4, scan, if_neg (by decide), innerCandidates315_4_checked]
  change scan edge (search edge 3) 4 [2111] 2111 innerTail315_5
    (if 3 ≤ innerCandidates315_4.length then search edge 3 [2111, 3083] innerCandidates315_4 5194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan315_5

def innerTail315_3 : List Nat := 3041 :: innerTail315_4
def innerCandidates315_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates315_3_checked : candidates edge 4 2111 3041 innerTail315_4 best.bound = innerCandidates315_3 := by
  unfold candidates
  have row : edge 3041 = SearchData.row3041.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan315_3 : scan edge (search edge 3) 4 [2111] 2111 innerTail315_3 best = best := by
  rw [innerTail315_3, scan, if_neg (by decide), innerCandidates315_3_checked]
  change scan edge (search edge 3) 4 [2111] 2111 innerTail315_4
    (if 3 ≤ innerCandidates315_3.length then search edge 3 [2111, 3041] innerCandidates315_3 5152 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan315_4

def innerTail315_2 : List Nat := 2999 :: innerTail315_3
def innerCandidates315_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates315_2_checked : candidates edge 4 2111 2999 innerTail315_3 best.bound = innerCandidates315_2 := by
  unfold candidates
  have row : edge 2999 = SearchData.row2999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan315_2 : scan edge (search edge 3) 4 [2111] 2111 innerTail315_2 best = best := by
  rw [innerTail315_2, scan, if_neg (by decide), innerCandidates315_2_checked]
  change scan edge (search edge 3) 4 [2111] 2111 innerTail315_3
    (if 3 ≤ innerCandidates315_2.length then search edge 3 [2111, 2999] innerCandidates315_2 5110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan315_3

def innerTail315_1 : List Nat := 2969 :: innerTail315_2
def innerCandidates315_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates315_1_checked : candidates edge 4 2111 2969 innerTail315_2 best.bound = innerCandidates315_1 := by
  unfold candidates
  have row : edge 2969 = SearchData.row2969.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan315_1 : scan edge (search edge 3) 4 [2111] 2111 innerTail315_1 best = best := by
  rw [innerTail315_1, scan, if_neg (by decide), innerCandidates315_1_checked]
  change scan edge (search edge 3) 4 [2111] 2111 innerTail315_2
    (if 3 ≤ innerCandidates315_1.length then search edge 3 [2111, 2969] innerCandidates315_1 5080 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan315_2

def innerTail315_0 : List Nat := 2411 :: innerTail315_1
def innerCandidates315_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates315_0_checked : candidates edge 4 2111 2411 innerTail315_1 best.bound = innerCandidates315_0 := by
  unfold candidates
  have row : edge 2411 = PrimePairCache.smallRow2411.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan315_0 : scan edge (search edge 3) 4 [2111] 2111 innerTail315_0 best = best := by
  rw [innerTail315_0, scan, if_neg (by decide), innerCandidates315_0_checked]
  change scan edge (search edge 3) 4 [2111] 2111 innerTail315_1
    (if 3 ≤ innerCandidates315_0.length then search edge 3 [2111, 2411] innerCandidates315_0 4522 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan315_1

theorem rootBranch315 : search edge 4 [2111] rootCandidates315 2111 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2111] 2111 innerTail315_0 best = best
  exact innerScan315_0

end ProjectEulerSolutions.PrimePairUpper.Search
