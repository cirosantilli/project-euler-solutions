import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates674 : List Nat := [5323, 5659, 6691, 7213, 8269]
set_option maxRecDepth 4096 in
theorem rootCandidates674_checked : candidates edge 5 0 5059 tail675 best.bound = rootCandidates674 := by
  unfold candidates
  have row : edge 5059 = SearchData.row5059.contains := by funext q; rfl
  rw [row]
  decide

def innerTail674_5 : List Nat := []
theorem innerScan674_5 : scan edge (search edge 3) 4 [5059] 5059 innerTail674_5 best = best := rfl

def innerTail674_4 : List Nat := 8269 :: innerTail674_5
def innerCandidates674_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates674_4_checked : candidates edge 4 5059 8269 innerTail674_5 best.bound = innerCandidates674_4 := by
  unfold candidates
  have row : edge 8269 = SearchData.row8269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan674_4 : scan edge (search edge 3) 4 [5059] 5059 innerTail674_4 best = best := by
  rw [innerTail674_4, scan, if_neg (by decide), innerCandidates674_4_checked]
  change scan edge (search edge 3) 4 [5059] 5059 innerTail674_5
    (if 3 ≤ innerCandidates674_4.length then search edge 3 [5059, 8269] innerCandidates674_4 13328 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan674_5

def innerTail674_3 : List Nat := 7213 :: innerTail674_4
def innerCandidates674_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates674_3_checked : candidates edge 4 5059 7213 innerTail674_4 best.bound = innerCandidates674_3 := by
  unfold candidates
  have row : edge 7213 = SearchData.row7213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan674_3 : scan edge (search edge 3) 4 [5059] 5059 innerTail674_3 best = best := by
  rw [innerTail674_3, scan, if_neg (by decide), innerCandidates674_3_checked]
  change scan edge (search edge 3) 4 [5059] 5059 innerTail674_4
    (if 3 ≤ innerCandidates674_3.length then search edge 3 [5059, 7213] innerCandidates674_3 12272 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan674_4

def innerTail674_2 : List Nat := 6691 :: innerTail674_3
def innerCandidates674_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates674_2_checked : candidates edge 4 5059 6691 innerTail674_3 best.bound = innerCandidates674_2 := by
  unfold candidates
  have row : edge 6691 = SearchData.row6691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan674_2 : scan edge (search edge 3) 4 [5059] 5059 innerTail674_2 best = best := by
  rw [innerTail674_2, scan, if_neg (by decide), innerCandidates674_2_checked]
  change scan edge (search edge 3) 4 [5059] 5059 innerTail674_3
    (if 3 ≤ innerCandidates674_2.length then search edge 3 [5059, 6691] innerCandidates674_2 11750 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan674_3

def innerTail674_1 : List Nat := 5659 :: innerTail674_2
def innerCandidates674_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates674_1_checked : candidates edge 4 5059 5659 innerTail674_2 best.bound = innerCandidates674_1 := by
  unfold candidates
  have row : edge 5659 = SearchData.row5659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan674_1 : scan edge (search edge 3) 4 [5059] 5059 innerTail674_1 best = best := by
  rw [innerTail674_1, scan, if_neg (by decide), innerCandidates674_1_checked]
  change scan edge (search edge 3) 4 [5059] 5059 innerTail674_2
    (if 3 ≤ innerCandidates674_1.length then search edge 3 [5059, 5659] innerCandidates674_1 10718 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan674_2

def innerTail674_0 : List Nat := 5323 :: innerTail674_1
def innerCandidates674_0 : List Nat := [5659]
set_option maxRecDepth 4096 in
theorem innerCandidates674_0_checked : candidates edge 4 5059 5323 innerTail674_1 best.bound = innerCandidates674_0 := by
  unfold candidates
  have row : edge 5323 = SearchData.row5323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan674_0 : scan edge (search edge 3) 4 [5059] 5059 innerTail674_0 best = best := by
  rw [innerTail674_0, scan, if_neg (by decide), innerCandidates674_0_checked]
  change scan edge (search edge 3) 4 [5059] 5059 innerTail674_1
    (if 3 ≤ innerCandidates674_0.length then search edge 3 [5059, 5323] innerCandidates674_0 10382 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan674_1

theorem rootBranch674 : search edge 4 [5059] rootCandidates674 5059 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5059] 5059 innerTail674_0 best = best
  exact innerScan674_0

def rootCandidates673 : List Nat := [5867, 5903, 5939, 6101, 6737, 6833, 6863, 7253, 7283, 7949]
set_option maxRecDepth 4096 in
theorem rootCandidates673_checked : candidates edge 5 0 5051 tail674 best.bound = rootCandidates673 := by
  unfold candidates
  have row : edge 5051 = SearchData.row5051.contains := by funext q; rfl
  rw [row]
  decide

def innerTail673_10 : List Nat := []
theorem innerScan673_10 : scan edge (search edge 3) 4 [5051] 5051 innerTail673_10 best = best := rfl

def innerTail673_9 : List Nat := 7949 :: innerTail673_10
def innerCandidates673_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates673_9_checked : candidates edge 4 5051 7949 innerTail673_10 best.bound = innerCandidates673_9 := by
  unfold candidates
  have row : edge 7949 = SearchData.row7949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan673_9 : scan edge (search edge 3) 4 [5051] 5051 innerTail673_9 best = best := by
  rw [innerTail673_9, scan, if_neg (by decide), innerCandidates673_9_checked]
  change scan edge (search edge 3) 4 [5051] 5051 innerTail673_10
    (if 3 ≤ innerCandidates673_9.length then search edge 3 [5051, 7949] innerCandidates673_9 13000 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan673_10

def innerTail673_8 : List Nat := 7283 :: innerTail673_9
def innerCandidates673_8 : List Nat := [7949]
set_option maxRecDepth 4096 in
theorem innerCandidates673_8_checked : candidates edge 4 5051 7283 innerTail673_9 best.bound = innerCandidates673_8 := by
  unfold candidates
  have row : edge 7283 = SearchData.row7283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan673_8 : scan edge (search edge 3) 4 [5051] 5051 innerTail673_8 best = best := by
  rw [innerTail673_8, scan, if_neg (by decide), innerCandidates673_8_checked]
  change scan edge (search edge 3) 4 [5051] 5051 innerTail673_9
    (if 3 ≤ innerCandidates673_8.length then search edge 3 [5051, 7283] innerCandidates673_8 12334 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan673_9

def innerTail673_7 : List Nat := 7253 :: innerTail673_8
def innerCandidates673_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates673_7_checked : candidates edge 4 5051 7253 innerTail673_8 best.bound = innerCandidates673_7 := by
  unfold candidates
  have row : edge 7253 = SearchData.row7253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan673_7 : scan edge (search edge 3) 4 [5051] 5051 innerTail673_7 best = best := by
  rw [innerTail673_7, scan, if_neg (by decide), innerCandidates673_7_checked]
  change scan edge (search edge 3) 4 [5051] 5051 innerTail673_8
    (if 3 ≤ innerCandidates673_7.length then search edge 3 [5051, 7253] innerCandidates673_7 12304 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan673_8

def innerTail673_6 : List Nat := 6863 :: innerTail673_7
def innerCandidates673_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates673_6_checked : candidates edge 4 5051 6863 innerTail673_7 best.bound = innerCandidates673_6 := by
  unfold candidates
  have row : edge 6863 = SearchData.row6863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan673_6 : scan edge (search edge 3) 4 [5051] 5051 innerTail673_6 best = best := by
  rw [innerTail673_6, scan, if_neg (by decide), innerCandidates673_6_checked]
  change scan edge (search edge 3) 4 [5051] 5051 innerTail673_7
    (if 3 ≤ innerCandidates673_6.length then search edge 3 [5051, 6863] innerCandidates673_6 11914 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan673_7

def innerTail673_5 : List Nat := 6833 :: innerTail673_6
def innerCandidates673_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates673_5_checked : candidates edge 4 5051 6833 innerTail673_6 best.bound = innerCandidates673_5 := by
  unfold candidates
  have row : edge 6833 = SearchData.row6833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan673_5 : scan edge (search edge 3) 4 [5051] 5051 innerTail673_5 best = best := by
  rw [innerTail673_5, scan, if_neg (by decide), innerCandidates673_5_checked]
  change scan edge (search edge 3) 4 [5051] 5051 innerTail673_6
    (if 3 ≤ innerCandidates673_5.length then search edge 3 [5051, 6833] innerCandidates673_5 11884 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan673_6

def innerTail673_4 : List Nat := 6737 :: innerTail673_5
def innerCandidates673_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates673_4_checked : candidates edge 4 5051 6737 innerTail673_5 best.bound = innerCandidates673_4 := by
  unfold candidates
  have row : edge 6737 = SearchData.row6737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan673_4 : scan edge (search edge 3) 4 [5051] 5051 innerTail673_4 best = best := by
  rw [innerTail673_4, scan, if_neg (by decide), innerCandidates673_4_checked]
  change scan edge (search edge 3) 4 [5051] 5051 innerTail673_5
    (if 3 ≤ innerCandidates673_4.length then search edge 3 [5051, 6737] innerCandidates673_4 11788 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan673_5

def innerTail673_3 : List Nat := 6101 :: innerTail673_4
def innerCandidates673_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates673_3_checked : candidates edge 4 5051 6101 innerTail673_4 best.bound = innerCandidates673_3 := by
  unfold candidates
  have row : edge 6101 = SearchData.row6101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan673_3 : scan edge (search edge 3) 4 [5051] 5051 innerTail673_3 best = best := by
  rw [innerTail673_3, scan, if_neg (by decide), innerCandidates673_3_checked]
  change scan edge (search edge 3) 4 [5051] 5051 innerTail673_4
    (if 3 ≤ innerCandidates673_3.length then search edge 3 [5051, 6101] innerCandidates673_3 11152 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan673_4

def innerTail673_2 : List Nat := 5939 :: innerTail673_3
def innerCandidates673_2 : List Nat := [6101]
set_option maxRecDepth 4096 in
theorem innerCandidates673_2_checked : candidates edge 4 5051 5939 innerTail673_3 best.bound = innerCandidates673_2 := by
  unfold candidates
  have row : edge 5939 = SearchData.row5939.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan673_2 : scan edge (search edge 3) 4 [5051] 5051 innerTail673_2 best = best := by
  rw [innerTail673_2, scan, if_neg (by decide), innerCandidates673_2_checked]
  change scan edge (search edge 3) 4 [5051] 5051 innerTail673_3
    (if 3 ≤ innerCandidates673_2.length then search edge 3 [5051, 5939] innerCandidates673_2 10990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan673_3

def innerTail673_1 : List Nat := 5903 :: innerTail673_2
def innerCandidates673_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates673_1_checked : candidates edge 4 5051 5903 innerTail673_2 best.bound = innerCandidates673_1 := by
  unfold candidates
  have row : edge 5903 = SearchData.row5903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan673_1 : scan edge (search edge 3) 4 [5051] 5051 innerTail673_1 best = best := by
  rw [innerTail673_1, scan, if_neg (by decide), innerCandidates673_1_checked]
  change scan edge (search edge 3) 4 [5051] 5051 innerTail673_2
    (if 3 ≤ innerCandidates673_1.length then search edge 3 [5051, 5903] innerCandidates673_1 10954 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan673_2

def innerTail673_0 : List Nat := 5867 :: innerTail673_1
def innerCandidates673_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates673_0_checked : candidates edge 4 5051 5867 innerTail673_1 best.bound = innerCandidates673_0 := by
  unfold candidates
  have row : edge 5867 = SearchData.row5867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan673_0 : scan edge (search edge 3) 4 [5051] 5051 innerTail673_0 best = best := by
  rw [innerTail673_0, scan, if_neg (by decide), innerCandidates673_0_checked]
  change scan edge (search edge 3) 4 [5051] 5051 innerTail673_1
    (if 3 ≤ innerCandidates673_0.length then search edge 3 [5051, 5867] innerCandidates673_0 10918 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan673_1

theorem rootBranch673 : search edge 4 [5051] rootCandidates673 5051 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5051] 5051 innerTail673_0 best = best
  exact innerScan673_0

def rootCandidates672 : List Nat := [5153, 5669, 6299, 6317, 6329, 7001, 7547, 7727, 8093, 8111]
set_option maxRecDepth 4096 in
theorem rootCandidates672_checked : candidates edge 5 0 5039 tail673 best.bound = rootCandidates672 := by
  unfold candidates
  have row : edge 5039 = SearchData.row5039.contains := by funext q; rfl
  rw [row]
  decide

def innerTail672_10 : List Nat := []
theorem innerScan672_10 : scan edge (search edge 3) 4 [5039] 5039 innerTail672_10 best = best := rfl

def innerTail672_9 : List Nat := 8111 :: innerTail672_10
def innerCandidates672_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates672_9_checked : candidates edge 4 5039 8111 innerTail672_10 best.bound = innerCandidates672_9 := by
  unfold candidates
  have row : edge 8111 = SearchData.row8111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan672_9 : scan edge (search edge 3) 4 [5039] 5039 innerTail672_9 best = best := by
  rw [innerTail672_9, scan, if_neg (by decide), innerCandidates672_9_checked]
  change scan edge (search edge 3) 4 [5039] 5039 innerTail672_10
    (if 3 ≤ innerCandidates672_9.length then search edge 3 [5039, 8111] innerCandidates672_9 13150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan672_10

def innerTail672_8 : List Nat := 8093 :: innerTail672_9
def innerCandidates672_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates672_8_checked : candidates edge 4 5039 8093 innerTail672_9 best.bound = innerCandidates672_8 := by
  unfold candidates
  have row : edge 8093 = SearchData.row8093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan672_8 : scan edge (search edge 3) 4 [5039] 5039 innerTail672_8 best = best := by
  rw [innerTail672_8, scan, if_neg (by decide), innerCandidates672_8_checked]
  change scan edge (search edge 3) 4 [5039] 5039 innerTail672_9
    (if 3 ≤ innerCandidates672_8.length then search edge 3 [5039, 8093] innerCandidates672_8 13132 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan672_9

def innerTail672_7 : List Nat := 7727 :: innerTail672_8
def innerCandidates672_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates672_7_checked : candidates edge 4 5039 7727 innerTail672_8 best.bound = innerCandidates672_7 := by
  unfold candidates
  have row : edge 7727 = SearchData.row7727.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan672_7 : scan edge (search edge 3) 4 [5039] 5039 innerTail672_7 best = best := by
  rw [innerTail672_7, scan, if_neg (by decide), innerCandidates672_7_checked]
  change scan edge (search edge 3) 4 [5039] 5039 innerTail672_8
    (if 3 ≤ innerCandidates672_7.length then search edge 3 [5039, 7727] innerCandidates672_7 12766 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan672_8

def innerTail672_6 : List Nat := 7547 :: innerTail672_7
def innerCandidates672_6 : List Nat := [8093]
set_option maxRecDepth 4096 in
theorem innerCandidates672_6_checked : candidates edge 4 5039 7547 innerTail672_7 best.bound = innerCandidates672_6 := by
  unfold candidates
  have row : edge 7547 = SearchData.row7547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan672_6 : scan edge (search edge 3) 4 [5039] 5039 innerTail672_6 best = best := by
  rw [innerTail672_6, scan, if_neg (by decide), innerCandidates672_6_checked]
  change scan edge (search edge 3) 4 [5039] 5039 innerTail672_7
    (if 3 ≤ innerCandidates672_6.length then search edge 3 [5039, 7547] innerCandidates672_6 12586 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan672_7

def innerTail672_5 : List Nat := 7001 :: innerTail672_6
def innerCandidates672_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates672_5_checked : candidates edge 4 5039 7001 innerTail672_6 best.bound = innerCandidates672_5 := by
  unfold candidates
  have row : edge 7001 = SearchData.row7001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan672_5 : scan edge (search edge 3) 4 [5039] 5039 innerTail672_5 best = best := by
  rw [innerTail672_5, scan, if_neg (by decide), innerCandidates672_5_checked]
  change scan edge (search edge 3) 4 [5039] 5039 innerTail672_6
    (if 3 ≤ innerCandidates672_5.length then search edge 3 [5039, 7001] innerCandidates672_5 12040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan672_6

def innerTail672_4 : List Nat := 6329 :: innerTail672_5
def innerCandidates672_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates672_4_checked : candidates edge 4 5039 6329 innerTail672_5 best.bound = innerCandidates672_4 := by
  unfold candidates
  have row : edge 6329 = SearchData.row6329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan672_4 : scan edge (search edge 3) 4 [5039] 5039 innerTail672_4 best = best := by
  rw [innerTail672_4, scan, if_neg (by decide), innerCandidates672_4_checked]
  change scan edge (search edge 3) 4 [5039] 5039 innerTail672_5
    (if 3 ≤ innerCandidates672_4.length then search edge 3 [5039, 6329] innerCandidates672_4 11368 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan672_5

def innerTail672_3 : List Nat := 6317 :: innerTail672_4
def innerCandidates672_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates672_3_checked : candidates edge 4 5039 6317 innerTail672_4 best.bound = innerCandidates672_3 := by
  unfold candidates
  have row : edge 6317 = SearchData.row6317.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan672_3 : scan edge (search edge 3) 4 [5039] 5039 innerTail672_3 best = best := by
  rw [innerTail672_3, scan, if_neg (by decide), innerCandidates672_3_checked]
  change scan edge (search edge 3) 4 [5039] 5039 innerTail672_4
    (if 3 ≤ innerCandidates672_3.length then search edge 3 [5039, 6317] innerCandidates672_3 11356 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan672_4

def innerTail672_2 : List Nat := 6299 :: innerTail672_3
def innerCandidates672_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates672_2_checked : candidates edge 4 5039 6299 innerTail672_3 best.bound = innerCandidates672_2 := by
  unfold candidates
  have row : edge 6299 = SearchData.row6299.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan672_2 : scan edge (search edge 3) 4 [5039] 5039 innerTail672_2 best = best := by
  rw [innerTail672_2, scan, if_neg (by decide), innerCandidates672_2_checked]
  change scan edge (search edge 3) 4 [5039] 5039 innerTail672_3
    (if 3 ≤ innerCandidates672_2.length then search edge 3 [5039, 6299] innerCandidates672_2 11338 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan672_3

def innerTail672_1 : List Nat := 5669 :: innerTail672_2
def innerCandidates672_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates672_1_checked : candidates edge 4 5039 5669 innerTail672_2 best.bound = innerCandidates672_1 := by
  unfold candidates
  have row : edge 5669 = SearchData.row5669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan672_1 : scan edge (search edge 3) 4 [5039] 5039 innerTail672_1 best = best := by
  rw [innerTail672_1, scan, if_neg (by decide), innerCandidates672_1_checked]
  change scan edge (search edge 3) 4 [5039] 5039 innerTail672_2
    (if 3 ≤ innerCandidates672_1.length then search edge 3 [5039, 5669] innerCandidates672_1 10708 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan672_2

def innerTail672_0 : List Nat := 5153 :: innerTail672_1
def innerCandidates672_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates672_0_checked : candidates edge 4 5039 5153 innerTail672_1 best.bound = innerCandidates672_0 := by
  unfold candidates
  have row : edge 5153 = SearchData.row5153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan672_0 : scan edge (search edge 3) 4 [5039] 5039 innerTail672_0 best = best := by
  rw [innerTail672_0, scan, if_neg (by decide), innerCandidates672_0_checked]
  change scan edge (search edge 3) 4 [5039] 5039 innerTail672_1
    (if 3 ≤ innerCandidates672_0.length then search edge 3 [5039, 5153] innerCandidates672_0 10192 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan672_1

theorem rootBranch672 : search edge 4 [5039] rootCandidates672 5039 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5039] 5039 innerTail672_0 best = best
  exact innerScan672_0

def rootCandidates671 : List Nat := [5443, 6841, 7039, 7477, 7753, 8059, 8269]
set_option maxRecDepth 4096 in
theorem rootCandidates671_checked : candidates edge 5 0 5023 tail672 best.bound = rootCandidates671 := by
  unfold candidates
  have row : edge 5023 = SearchData.row5023.contains := by funext q; rfl
  rw [row]
  decide

def innerTail671_7 : List Nat := []
theorem innerScan671_7 : scan edge (search edge 3) 4 [5023] 5023 innerTail671_7 best = best := rfl

def innerTail671_6 : List Nat := 8269 :: innerTail671_7
def innerCandidates671_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates671_6_checked : candidates edge 4 5023 8269 innerTail671_7 best.bound = innerCandidates671_6 := by
  unfold candidates
  have row : edge 8269 = SearchData.row8269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan671_6 : scan edge (search edge 3) 4 [5023] 5023 innerTail671_6 best = best := by
  rw [innerTail671_6, scan, if_neg (by decide), innerCandidates671_6_checked]
  change scan edge (search edge 3) 4 [5023] 5023 innerTail671_7
    (if 3 ≤ innerCandidates671_6.length then search edge 3 [5023, 8269] innerCandidates671_6 13292 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan671_7

def innerTail671_5 : List Nat := 8059 :: innerTail671_6
def innerCandidates671_5 : List Nat := [8269]
set_option maxRecDepth 4096 in
theorem innerCandidates671_5_checked : candidates edge 4 5023 8059 innerTail671_6 best.bound = innerCandidates671_5 := by
  unfold candidates
  have row : edge 8059 = SearchData.row8059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan671_5 : scan edge (search edge 3) 4 [5023] 5023 innerTail671_5 best = best := by
  rw [innerTail671_5, scan, if_neg (by decide), innerCandidates671_5_checked]
  change scan edge (search edge 3) 4 [5023] 5023 innerTail671_6
    (if 3 ≤ innerCandidates671_5.length then search edge 3 [5023, 8059] innerCandidates671_5 13082 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan671_6

def innerTail671_4 : List Nat := 7753 :: innerTail671_5
def innerCandidates671_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates671_4_checked : candidates edge 4 5023 7753 innerTail671_5 best.bound = innerCandidates671_4 := by
  unfold candidates
  have row : edge 7753 = SearchData.row7753.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan671_4 : scan edge (search edge 3) 4 [5023] 5023 innerTail671_4 best = best := by
  rw [innerTail671_4, scan, if_neg (by decide), innerCandidates671_4_checked]
  change scan edge (search edge 3) 4 [5023] 5023 innerTail671_5
    (if 3 ≤ innerCandidates671_4.length then search edge 3 [5023, 7753] innerCandidates671_4 12776 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan671_5

def innerTail671_3 : List Nat := 7477 :: innerTail671_4
def innerCandidates671_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates671_3_checked : candidates edge 4 5023 7477 innerTail671_4 best.bound = innerCandidates671_3 := by
  unfold candidates
  have row : edge 7477 = SearchData.row7477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan671_3 : scan edge (search edge 3) 4 [5023] 5023 innerTail671_3 best = best := by
  rw [innerTail671_3, scan, if_neg (by decide), innerCandidates671_3_checked]
  change scan edge (search edge 3) 4 [5023] 5023 innerTail671_4
    (if 3 ≤ innerCandidates671_3.length then search edge 3 [5023, 7477] innerCandidates671_3 12500 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan671_4

def innerTail671_2 : List Nat := 7039 :: innerTail671_3
def innerCandidates671_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates671_2_checked : candidates edge 4 5023 7039 innerTail671_3 best.bound = innerCandidates671_2 := by
  unfold candidates
  have row : edge 7039 = SearchData.row7039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan671_2 : scan edge (search edge 3) 4 [5023] 5023 innerTail671_2 best = best := by
  rw [innerTail671_2, scan, if_neg (by decide), innerCandidates671_2_checked]
  change scan edge (search edge 3) 4 [5023] 5023 innerTail671_3
    (if 3 ≤ innerCandidates671_2.length then search edge 3 [5023, 7039] innerCandidates671_2 12062 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan671_3

def innerTail671_1 : List Nat := 6841 :: innerTail671_2
def innerCandidates671_1 : List Nat := [7039]
set_option maxRecDepth 4096 in
theorem innerCandidates671_1_checked : candidates edge 4 5023 6841 innerTail671_2 best.bound = innerCandidates671_1 := by
  unfold candidates
  have row : edge 6841 = SearchData.row6841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan671_1 : scan edge (search edge 3) 4 [5023] 5023 innerTail671_1 best = best := by
  rw [innerTail671_1, scan, if_neg (by decide), innerCandidates671_1_checked]
  change scan edge (search edge 3) 4 [5023] 5023 innerTail671_2
    (if 3 ≤ innerCandidates671_1.length then search edge 3 [5023, 6841] innerCandidates671_1 11864 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan671_2

def innerTail671_0 : List Nat := 5443 :: innerTail671_1
def innerCandidates671_0 : List Nat := [6841, 7039]
set_option maxRecDepth 4096 in
theorem innerCandidates671_0_checked : candidates edge 4 5023 5443 innerTail671_1 best.bound = innerCandidates671_0 := by
  unfold candidates
  have row : edge 5443 = SearchData.row5443.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan671_0 : scan edge (search edge 3) 4 [5023] 5023 innerTail671_0 best = best := by
  rw [innerTail671_0, scan, if_neg (by decide), innerCandidates671_0_checked]
  change scan edge (search edge 3) 4 [5023] 5023 innerTail671_1
    (if 3 ≤ innerCandidates671_0.length then search edge 3 [5023, 5443] innerCandidates671_0 10466 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan671_1

theorem rootBranch671 : search edge 4 [5023] rootCandidates671 5023 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5023] 5023 innerTail671_0 best = best
  exact innerScan671_0

def rootCandidates670 : List Nat := [5507, 5741, 5987, 6053, 7127, 8219]
set_option maxRecDepth 4096 in
theorem rootCandidates670_checked : candidates edge 5 0 5021 tail671 best.bound = rootCandidates670 := by
  unfold candidates
  have row : edge 5021 = SearchData.row5021.contains := by funext q; rfl
  rw [row]
  decide

def innerTail670_6 : List Nat := []
theorem innerScan670_6 : scan edge (search edge 3) 4 [5021] 5021 innerTail670_6 best = best := rfl

def innerTail670_5 : List Nat := 8219 :: innerTail670_6
def innerCandidates670_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates670_5_checked : candidates edge 4 5021 8219 innerTail670_6 best.bound = innerCandidates670_5 := by
  unfold candidates
  have row : edge 8219 = SearchData.row8219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan670_5 : scan edge (search edge 3) 4 [5021] 5021 innerTail670_5 best = best := by
  rw [innerTail670_5, scan, if_neg (by decide), innerCandidates670_5_checked]
  change scan edge (search edge 3) 4 [5021] 5021 innerTail670_6
    (if 3 ≤ innerCandidates670_5.length then search edge 3 [5021, 8219] innerCandidates670_5 13240 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan670_6

def innerTail670_4 : List Nat := 7127 :: innerTail670_5
def innerCandidates670_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates670_4_checked : candidates edge 4 5021 7127 innerTail670_5 best.bound = innerCandidates670_4 := by
  unfold candidates
  have row : edge 7127 = SearchData.row7127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan670_4 : scan edge (search edge 3) 4 [5021] 5021 innerTail670_4 best = best := by
  rw [innerTail670_4, scan, if_neg (by decide), innerCandidates670_4_checked]
  change scan edge (search edge 3) 4 [5021] 5021 innerTail670_5
    (if 3 ≤ innerCandidates670_4.length then search edge 3 [5021, 7127] innerCandidates670_4 12148 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan670_5

def innerTail670_3 : List Nat := 6053 :: innerTail670_4
def innerCandidates670_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates670_3_checked : candidates edge 4 5021 6053 innerTail670_4 best.bound = innerCandidates670_3 := by
  unfold candidates
  have row : edge 6053 = SearchData.row6053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan670_3 : scan edge (search edge 3) 4 [5021] 5021 innerTail670_3 best = best := by
  rw [innerTail670_3, scan, if_neg (by decide), innerCandidates670_3_checked]
  change scan edge (search edge 3) 4 [5021] 5021 innerTail670_4
    (if 3 ≤ innerCandidates670_3.length then search edge 3 [5021, 6053] innerCandidates670_3 11074 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan670_4

def innerTail670_2 : List Nat := 5987 :: innerTail670_3
def innerCandidates670_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates670_2_checked : candidates edge 4 5021 5987 innerTail670_3 best.bound = innerCandidates670_2 := by
  unfold candidates
  have row : edge 5987 = SearchData.row5987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan670_2 : scan edge (search edge 3) 4 [5021] 5021 innerTail670_2 best = best := by
  rw [innerTail670_2, scan, if_neg (by decide), innerCandidates670_2_checked]
  change scan edge (search edge 3) 4 [5021] 5021 innerTail670_3
    (if 3 ≤ innerCandidates670_2.length then search edge 3 [5021, 5987] innerCandidates670_2 11008 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan670_3

def innerTail670_1 : List Nat := 5741 :: innerTail670_2
def innerCandidates670_1 : List Nat := [8219]
set_option maxRecDepth 4096 in
theorem innerCandidates670_1_checked : candidates edge 4 5021 5741 innerTail670_2 best.bound = innerCandidates670_1 := by
  unfold candidates
  have row : edge 5741 = SearchData.row5741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan670_1 : scan edge (search edge 3) 4 [5021] 5021 innerTail670_1 best = best := by
  rw [innerTail670_1, scan, if_neg (by decide), innerCandidates670_1_checked]
  change scan edge (search edge 3) 4 [5021] 5021 innerTail670_2
    (if 3 ≤ innerCandidates670_1.length then search edge 3 [5021, 5741] innerCandidates670_1 10762 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan670_2

def innerTail670_0 : List Nat := 5507 :: innerTail670_1
def innerCandidates670_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates670_0_checked : candidates edge 4 5021 5507 innerTail670_1 best.bound = innerCandidates670_0 := by
  unfold candidates
  have row : edge 5507 = SearchData.row5507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan670_0 : scan edge (search edge 3) 4 [5021] 5021 innerTail670_0 best = best := by
  rw [innerTail670_0, scan, if_neg (by decide), innerCandidates670_0_checked]
  change scan edge (search edge 3) 4 [5021] 5021 innerTail670_1
    (if 3 ≤ innerCandidates670_0.length then search edge 3 [5021, 5507] innerCandidates670_0 10528 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan670_1

theorem rootBranch670 : search edge 4 [5021] rootCandidates670 5021 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5021] 5021 innerTail670_0 best = best
  exact innerScan670_0

def rootCandidates669 : List Nat := [5209, 6163, 6271, 6397, 6949, 6961, 7393, 8053, 8191]
set_option maxRecDepth 4096 in
theorem rootCandidates669_checked : candidates edge 5 0 5011 tail670 best.bound = rootCandidates669 := by
  unfold candidates
  have row : edge 5011 = SearchData.row5011.contains := by funext q; rfl
  rw [row]
  decide

def innerTail669_9 : List Nat := []
theorem innerScan669_9 : scan edge (search edge 3) 4 [5011] 5011 innerTail669_9 best = best := rfl

def innerTail669_8 : List Nat := 8191 :: innerTail669_9
def innerCandidates669_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates669_8_checked : candidates edge 4 5011 8191 innerTail669_9 best.bound = innerCandidates669_8 := by
  unfold candidates
  have row : edge 8191 = SearchData.row8191.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan669_8 : scan edge (search edge 3) 4 [5011] 5011 innerTail669_8 best = best := by
  rw [innerTail669_8, scan, if_neg (by decide), innerCandidates669_8_checked]
  change scan edge (search edge 3) 4 [5011] 5011 innerTail669_9
    (if 3 ≤ innerCandidates669_8.length then search edge 3 [5011, 8191] innerCandidates669_8 13202 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan669_9

def innerTail669_7 : List Nat := 8053 :: innerTail669_8
def innerCandidates669_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates669_7_checked : candidates edge 4 5011 8053 innerTail669_8 best.bound = innerCandidates669_7 := by
  unfold candidates
  have row : edge 8053 = SearchData.row8053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan669_7 : scan edge (search edge 3) 4 [5011] 5011 innerTail669_7 best = best := by
  rw [innerTail669_7, scan, if_neg (by decide), innerCandidates669_7_checked]
  change scan edge (search edge 3) 4 [5011] 5011 innerTail669_8
    (if 3 ≤ innerCandidates669_7.length then search edge 3 [5011, 8053] innerCandidates669_7 13064 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan669_8

def innerTail669_6 : List Nat := 7393 :: innerTail669_7
def innerCandidates669_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates669_6_checked : candidates edge 4 5011 7393 innerTail669_7 best.bound = innerCandidates669_6 := by
  unfold candidates
  have row : edge 7393 = SearchData.row7393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan669_6 : scan edge (search edge 3) 4 [5011] 5011 innerTail669_6 best = best := by
  rw [innerTail669_6, scan, if_neg (by decide), innerCandidates669_6_checked]
  change scan edge (search edge 3) 4 [5011] 5011 innerTail669_7
    (if 3 ≤ innerCandidates669_6.length then search edge 3 [5011, 7393] innerCandidates669_6 12404 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan669_7

def innerTail669_5 : List Nat := 6961 :: innerTail669_6
def innerCandidates669_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates669_5_checked : candidates edge 4 5011 6961 innerTail669_6 best.bound = innerCandidates669_5 := by
  unfold candidates
  have row : edge 6961 = SearchData.row6961.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan669_5 : scan edge (search edge 3) 4 [5011] 5011 innerTail669_5 best = best := by
  rw [innerTail669_5, scan, if_neg (by decide), innerCandidates669_5_checked]
  change scan edge (search edge 3) 4 [5011] 5011 innerTail669_6
    (if 3 ≤ innerCandidates669_5.length then search edge 3 [5011, 6961] innerCandidates669_5 11972 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan669_6

def innerTail669_4 : List Nat := 6949 :: innerTail669_5
def innerCandidates669_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates669_4_checked : candidates edge 4 5011 6949 innerTail669_5 best.bound = innerCandidates669_4 := by
  unfold candidates
  have row : edge 6949 = SearchData.row6949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan669_4 : scan edge (search edge 3) 4 [5011] 5011 innerTail669_4 best = best := by
  rw [innerTail669_4, scan, if_neg (by decide), innerCandidates669_4_checked]
  change scan edge (search edge 3) 4 [5011] 5011 innerTail669_5
    (if 3 ≤ innerCandidates669_4.length then search edge 3 [5011, 6949] innerCandidates669_4 11960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan669_5

def innerTail669_3 : List Nat := 6397 :: innerTail669_4
def innerCandidates669_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates669_3_checked : candidates edge 4 5011 6397 innerTail669_4 best.bound = innerCandidates669_3 := by
  unfold candidates
  have row : edge 6397 = SearchData.row6397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan669_3 : scan edge (search edge 3) 4 [5011] 5011 innerTail669_3 best = best := by
  rw [innerTail669_3, scan, if_neg (by decide), innerCandidates669_3_checked]
  change scan edge (search edge 3) 4 [5011] 5011 innerTail669_4
    (if 3 ≤ innerCandidates669_3.length then search edge 3 [5011, 6397] innerCandidates669_3 11408 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan669_4

def innerTail669_2 : List Nat := 6271 :: innerTail669_3
def innerCandidates669_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates669_2_checked : candidates edge 4 5011 6271 innerTail669_3 best.bound = innerCandidates669_2 := by
  unfold candidates
  have row : edge 6271 = SearchData.row6271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan669_2 : scan edge (search edge 3) 4 [5011] 5011 innerTail669_2 best = best := by
  rw [innerTail669_2, scan, if_neg (by decide), innerCandidates669_2_checked]
  change scan edge (search edge 3) 4 [5011] 5011 innerTail669_3
    (if 3 ≤ innerCandidates669_2.length then search edge 3 [5011, 6271] innerCandidates669_2 11282 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan669_3

def innerTail669_1 : List Nat := 6163 :: innerTail669_2
def innerCandidates669_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates669_1_checked : candidates edge 4 5011 6163 innerTail669_2 best.bound = innerCandidates669_1 := by
  unfold candidates
  have row : edge 6163 = SearchData.row6163.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan669_1 : scan edge (search edge 3) 4 [5011] 5011 innerTail669_1 best = best := by
  rw [innerTail669_1, scan, if_neg (by decide), innerCandidates669_1_checked]
  change scan edge (search edge 3) 4 [5011] 5011 innerTail669_2
    (if 3 ≤ innerCandidates669_1.length then search edge 3 [5011, 6163] innerCandidates669_1 11174 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan669_2

def innerTail669_0 : List Nat := 5209 :: innerTail669_1
def innerCandidates669_0 : List Nat := [6397, 8191]
set_option maxRecDepth 4096 in
theorem innerCandidates669_0_checked : candidates edge 4 5011 5209 innerTail669_1 best.bound = innerCandidates669_0 := by
  unfold candidates
  have row : edge 5209 = SearchData.row5209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan669_0 : scan edge (search edge 3) 4 [5011] 5011 innerTail669_0 best = best := by
  rw [innerTail669_0, scan, if_neg (by decide), innerCandidates669_0_checked]
  change scan edge (search edge 3) 4 [5011] 5011 innerTail669_1
    (if 3 ≤ innerCandidates669_0.length then search edge 3 [5011, 5209] innerCandidates669_0 10220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan669_1

theorem rootBranch669 : search edge 4 [5011] rootCandidates669 5011 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5011] 5011 innerTail669_0 best = best
  exact innerScan669_0

def rootCandidates668 : List Nat := [5351, 5849, 5861, 6029, 6197, 6353, 6947, 7109, 7757, 8093]
set_option maxRecDepth 4096 in
theorem rootCandidates668_checked : candidates edge 5 0 5009 tail669 best.bound = rootCandidates668 := by
  unfold candidates
  have row : edge 5009 = SearchData.row5009.contains := by funext q; rfl
  rw [row]
  decide

def innerTail668_10 : List Nat := []
theorem innerScan668_10 : scan edge (search edge 3) 4 [5009] 5009 innerTail668_10 best = best := rfl

def innerTail668_9 : List Nat := 8093 :: innerTail668_10
def innerCandidates668_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates668_9_checked : candidates edge 4 5009 8093 innerTail668_10 best.bound = innerCandidates668_9 := by
  unfold candidates
  have row : edge 8093 = SearchData.row8093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan668_9 : scan edge (search edge 3) 4 [5009] 5009 innerTail668_9 best = best := by
  rw [innerTail668_9, scan, if_neg (by decide), innerCandidates668_9_checked]
  change scan edge (search edge 3) 4 [5009] 5009 innerTail668_10
    (if 3 ≤ innerCandidates668_9.length then search edge 3 [5009, 8093] innerCandidates668_9 13102 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan668_10

def innerTail668_8 : List Nat := 7757 :: innerTail668_9
def innerCandidates668_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates668_8_checked : candidates edge 4 5009 7757 innerTail668_9 best.bound = innerCandidates668_8 := by
  unfold candidates
  have row : edge 7757 = SearchData.row7757.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan668_8 : scan edge (search edge 3) 4 [5009] 5009 innerTail668_8 best = best := by
  rw [innerTail668_8, scan, if_neg (by decide), innerCandidates668_8_checked]
  change scan edge (search edge 3) 4 [5009] 5009 innerTail668_9
    (if 3 ≤ innerCandidates668_8.length then search edge 3 [5009, 7757] innerCandidates668_8 12766 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan668_9

def innerTail668_7 : List Nat := 7109 :: innerTail668_8
def innerCandidates668_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates668_7_checked : candidates edge 4 5009 7109 innerTail668_8 best.bound = innerCandidates668_7 := by
  unfold candidates
  have row : edge 7109 = SearchData.row7109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan668_7 : scan edge (search edge 3) 4 [5009] 5009 innerTail668_7 best = best := by
  rw [innerTail668_7, scan, if_neg (by decide), innerCandidates668_7_checked]
  change scan edge (search edge 3) 4 [5009] 5009 innerTail668_8
    (if 3 ≤ innerCandidates668_7.length then search edge 3 [5009, 7109] innerCandidates668_7 12118 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan668_8

def innerTail668_6 : List Nat := 6947 :: innerTail668_7
def innerCandidates668_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates668_6_checked : candidates edge 4 5009 6947 innerTail668_7 best.bound = innerCandidates668_6 := by
  unfold candidates
  have row : edge 6947 = SearchData.row6947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan668_6 : scan edge (search edge 3) 4 [5009] 5009 innerTail668_6 best = best := by
  rw [innerTail668_6, scan, if_neg (by decide), innerCandidates668_6_checked]
  change scan edge (search edge 3) 4 [5009] 5009 innerTail668_7
    (if 3 ≤ innerCandidates668_6.length then search edge 3 [5009, 6947] innerCandidates668_6 11956 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan668_7

def innerTail668_5 : List Nat := 6353 :: innerTail668_6
def innerCandidates668_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates668_5_checked : candidates edge 4 5009 6353 innerTail668_6 best.bound = innerCandidates668_5 := by
  unfold candidates
  have row : edge 6353 = SearchData.row6353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan668_5 : scan edge (search edge 3) 4 [5009] 5009 innerTail668_5 best = best := by
  rw [innerTail668_5, scan, if_neg (by decide), innerCandidates668_5_checked]
  change scan edge (search edge 3) 4 [5009] 5009 innerTail668_6
    (if 3 ≤ innerCandidates668_5.length then search edge 3 [5009, 6353] innerCandidates668_5 11362 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan668_6

def innerTail668_4 : List Nat := 6197 :: innerTail668_5
def innerCandidates668_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates668_4_checked : candidates edge 4 5009 6197 innerTail668_5 best.bound = innerCandidates668_4 := by
  unfold candidates
  have row : edge 6197 = SearchData.row6197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan668_4 : scan edge (search edge 3) 4 [5009] 5009 innerTail668_4 best = best := by
  rw [innerTail668_4, scan, if_neg (by decide), innerCandidates668_4_checked]
  change scan edge (search edge 3) 4 [5009] 5009 innerTail668_5
    (if 3 ≤ innerCandidates668_4.length then search edge 3 [5009, 6197] innerCandidates668_4 11206 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan668_5

def innerTail668_3 : List Nat := 6029 :: innerTail668_4
def innerCandidates668_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates668_3_checked : candidates edge 4 5009 6029 innerTail668_4 best.bound = innerCandidates668_3 := by
  unfold candidates
  have row : edge 6029 = SearchData.row6029.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan668_3 : scan edge (search edge 3) 4 [5009] 5009 innerTail668_3 best = best := by
  rw [innerTail668_3, scan, if_neg (by decide), innerCandidates668_3_checked]
  change scan edge (search edge 3) 4 [5009] 5009 innerTail668_4
    (if 3 ≤ innerCandidates668_3.length then search edge 3 [5009, 6029] innerCandidates668_3 11038 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan668_4

def innerTail668_2 : List Nat := 5861 :: innerTail668_3
def innerCandidates668_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates668_2_checked : candidates edge 4 5009 5861 innerTail668_3 best.bound = innerCandidates668_2 := by
  unfold candidates
  have row : edge 5861 = SearchData.row5861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan668_2 : scan edge (search edge 3) 4 [5009] 5009 innerTail668_2 best = best := by
  rw [innerTail668_2, scan, if_neg (by decide), innerCandidates668_2_checked]
  change scan edge (search edge 3) 4 [5009] 5009 innerTail668_3
    (if 3 ≤ innerCandidates668_2.length then search edge 3 [5009, 5861] innerCandidates668_2 10870 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan668_3

def innerTail668_1 : List Nat := 5849 :: innerTail668_2
def innerCandidates668_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates668_1_checked : candidates edge 4 5009 5849 innerTail668_2 best.bound = innerCandidates668_1 := by
  unfold candidates
  have row : edge 5849 = SearchData.row5849.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan668_1 : scan edge (search edge 3) 4 [5009] 5009 innerTail668_1 best = best := by
  rw [innerTail668_1, scan, if_neg (by decide), innerCandidates668_1_checked]
  change scan edge (search edge 3) 4 [5009] 5009 innerTail668_2
    (if 3 ≤ innerCandidates668_1.length then search edge 3 [5009, 5849] innerCandidates668_1 10858 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan668_2

def innerTail668_0 : List Nat := 5351 :: innerTail668_1
def innerCandidates668_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates668_0_checked : candidates edge 4 5009 5351 innerTail668_1 best.bound = innerCandidates668_0 := by
  unfold candidates
  have row : edge 5351 = SearchData.row5351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan668_0 : scan edge (search edge 3) 4 [5009] 5009 innerTail668_0 best = best := by
  rw [innerTail668_0, scan, if_neg (by decide), innerCandidates668_0_checked]
  change scan edge (search edge 3) 4 [5009] 5009 innerTail668_1
    (if 3 ≤ innerCandidates668_0.length then search edge 3 [5009, 5351] innerCandidates668_0 10360 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan668_1

theorem rootBranch668 : search edge 4 [5009] rootCandidates668 5009 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5009] 5009 innerTail668_0 best = best
  exact innerScan668_0

end ProjectEulerSolutions.PrimePairUpper.Search
