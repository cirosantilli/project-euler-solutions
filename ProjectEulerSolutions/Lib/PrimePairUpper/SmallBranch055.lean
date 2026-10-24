import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates618 : List Nat := [4651, 5101, 5197, 6121, 7213, 7369, 7621, 8167, 8287, 8293, 8329]
set_option maxRecDepth 4096 in
theorem rootCandidates618_checked : candidates edge 5 0 4591 tail619 best.bound = rootCandidates618 := by
  unfold candidates
  have row : edge 4591 = SearchData.row4591.contains := by funext q; rfl
  rw [row]
  decide

def innerTail618_11 : List Nat := []
theorem innerScan618_11 : scan edge (search edge 3) 4 [4591] 4591 innerTail618_11 best = best := rfl

def innerTail618_10 : List Nat := 8329 :: innerTail618_11
def innerCandidates618_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates618_10_checked : candidates edge 4 4591 8329 innerTail618_11 best.bound = innerCandidates618_10 := by
  unfold candidates
  have row : edge 8329 = SearchData.row8329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan618_10 : scan edge (search edge 3) 4 [4591] 4591 innerTail618_10 best = best := by
  rw [innerTail618_10, scan, if_neg (by decide), innerCandidates618_10_checked]
  change scan edge (search edge 3) 4 [4591] 4591 innerTail618_11
    (if 3 ≤ innerCandidates618_10.length then search edge 3 [4591, 8329] innerCandidates618_10 12920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan618_11

def innerTail618_9 : List Nat := 8293 :: innerTail618_10
def innerCandidates618_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates618_9_checked : candidates edge 4 4591 8293 innerTail618_10 best.bound = innerCandidates618_9 := by
  unfold candidates
  have row : edge 8293 = SearchData.row8293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan618_9 : scan edge (search edge 3) 4 [4591] 4591 innerTail618_9 best = best := by
  rw [innerTail618_9, scan, if_neg (by decide), innerCandidates618_9_checked]
  change scan edge (search edge 3) 4 [4591] 4591 innerTail618_10
    (if 3 ≤ innerCandidates618_9.length then search edge 3 [4591, 8293] innerCandidates618_9 12884 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan618_10

def innerTail618_8 : List Nat := 8287 :: innerTail618_9
def innerCandidates618_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates618_8_checked : candidates edge 4 4591 8287 innerTail618_9 best.bound = innerCandidates618_8 := by
  unfold candidates
  have row : edge 8287 = SearchData.row8287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan618_8 : scan edge (search edge 3) 4 [4591] 4591 innerTail618_8 best = best := by
  rw [innerTail618_8, scan, if_neg (by decide), innerCandidates618_8_checked]
  change scan edge (search edge 3) 4 [4591] 4591 innerTail618_9
    (if 3 ≤ innerCandidates618_8.length then search edge 3 [4591, 8287] innerCandidates618_8 12878 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan618_9

def innerTail618_7 : List Nat := 8167 :: innerTail618_8
def innerCandidates618_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates618_7_checked : candidates edge 4 4591 8167 innerTail618_8 best.bound = innerCandidates618_7 := by
  unfold candidates
  have row : edge 8167 = SearchData.row8167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan618_7 : scan edge (search edge 3) 4 [4591] 4591 innerTail618_7 best = best := by
  rw [innerTail618_7, scan, if_neg (by decide), innerCandidates618_7_checked]
  change scan edge (search edge 3) 4 [4591] 4591 innerTail618_8
    (if 3 ≤ innerCandidates618_7.length then search edge 3 [4591, 8167] innerCandidates618_7 12758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan618_8

def innerTail618_6 : List Nat := 7621 :: innerTail618_7
def innerCandidates618_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates618_6_checked : candidates edge 4 4591 7621 innerTail618_7 best.bound = innerCandidates618_6 := by
  unfold candidates
  have row : edge 7621 = SearchData.row7621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan618_6 : scan edge (search edge 3) 4 [4591] 4591 innerTail618_6 best = best := by
  rw [innerTail618_6, scan, if_neg (by decide), innerCandidates618_6_checked]
  change scan edge (search edge 3) 4 [4591] 4591 innerTail618_7
    (if 3 ≤ innerCandidates618_6.length then search edge 3 [4591, 7621] innerCandidates618_6 12212 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan618_7

def innerTail618_5 : List Nat := 7369 :: innerTail618_6
def innerCandidates618_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates618_5_checked : candidates edge 4 4591 7369 innerTail618_6 best.bound = innerCandidates618_5 := by
  unfold candidates
  have row : edge 7369 = SearchData.row7369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan618_5 : scan edge (search edge 3) 4 [4591] 4591 innerTail618_5 best = best := by
  rw [innerTail618_5, scan, if_neg (by decide), innerCandidates618_5_checked]
  change scan edge (search edge 3) 4 [4591] 4591 innerTail618_6
    (if 3 ≤ innerCandidates618_5.length then search edge 3 [4591, 7369] innerCandidates618_5 11960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan618_6

def innerTail618_4 : List Nat := 7213 :: innerTail618_5
def innerCandidates618_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates618_4_checked : candidates edge 4 4591 7213 innerTail618_5 best.bound = innerCandidates618_4 := by
  unfold candidates
  have row : edge 7213 = SearchData.row7213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan618_4 : scan edge (search edge 3) 4 [4591] 4591 innerTail618_4 best = best := by
  rw [innerTail618_4, scan, if_neg (by decide), innerCandidates618_4_checked]
  change scan edge (search edge 3) 4 [4591] 4591 innerTail618_5
    (if 3 ≤ innerCandidates618_4.length then search edge 3 [4591, 7213] innerCandidates618_4 11804 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan618_5

def innerTail618_3 : List Nat := 6121 :: innerTail618_4
def innerCandidates618_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates618_3_checked : candidates edge 4 4591 6121 innerTail618_4 best.bound = innerCandidates618_3 := by
  unfold candidates
  have row : edge 6121 = SearchData.row6121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan618_3 : scan edge (search edge 3) 4 [4591] 4591 innerTail618_3 best = best := by
  rw [innerTail618_3, scan, if_neg (by decide), innerCandidates618_3_checked]
  change scan edge (search edge 3) 4 [4591] 4591 innerTail618_4
    (if 3 ≤ innerCandidates618_3.length then search edge 3 [4591, 6121] innerCandidates618_3 10712 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan618_4

def innerTail618_2 : List Nat := 5197 :: innerTail618_3
def innerCandidates618_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates618_2_checked : candidates edge 4 4591 5197 innerTail618_3 best.bound = innerCandidates618_2 := by
  unfold candidates
  have row : edge 5197 = SearchData.row5197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan618_2 : scan edge (search edge 3) 4 [4591] 4591 innerTail618_2 best = best := by
  rw [innerTail618_2, scan, if_neg (by decide), innerCandidates618_2_checked]
  change scan edge (search edge 3) 4 [4591] 4591 innerTail618_3
    (if 3 ≤ innerCandidates618_2.length then search edge 3 [4591, 5197] innerCandidates618_2 9788 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan618_3

def innerTail618_1 : List Nat := 5101 :: innerTail618_2
def innerCandidates618_1 : List Nat := [8293]
set_option maxRecDepth 4096 in
theorem innerCandidates618_1_checked : candidates edge 4 4591 5101 innerTail618_2 best.bound = innerCandidates618_1 := by
  unfold candidates
  have row : edge 5101 = SearchData.row5101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan618_1 : scan edge (search edge 3) 4 [4591] 4591 innerTail618_1 best = best := by
  rw [innerTail618_1, scan, if_neg (by decide), innerCandidates618_1_checked]
  change scan edge (search edge 3) 4 [4591] 4591 innerTail618_2
    (if 3 ≤ innerCandidates618_1.length then search edge 3 [4591, 5101] innerCandidates618_1 9692 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan618_2

def innerTail618_0 : List Nat := 4651 :: innerTail618_1
def innerCandidates618_0 : List Nat := [8293]
set_option maxRecDepth 4096 in
theorem innerCandidates618_0_checked : candidates edge 4 4591 4651 innerTail618_1 best.bound = innerCandidates618_0 := by
  unfold candidates
  have row : edge 4651 = SearchData.row4651.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan618_0 : scan edge (search edge 3) 4 [4591] 4591 innerTail618_0 best = best := by
  rw [innerTail618_0, scan, if_neg (by decide), innerCandidates618_0_checked]
  change scan edge (search edge 3) 4 [4591] 4591 innerTail618_1
    (if 3 ≤ innerCandidates618_0.length then search edge 3 [4591, 4651] innerCandidates618_0 9242 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan618_1

theorem rootBranch618 : search edge 4 [4591] rootCandidates618 4591 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4591] 4591 innerTail618_0 best = best
  exact innerScan618_0

def rootCandidates617 : List Nat := [4637, 4787, 4871, 5003, 5039, 5651, 7523, 7577]
set_option maxRecDepth 4096 in
theorem rootCandidates617_checked : candidates edge 5 0 4583 tail618 best.bound = rootCandidates617 := by
  unfold candidates
  have row : edge 4583 = SearchData.row4583.contains := by funext q; rfl
  rw [row]
  decide

def innerTail617_8 : List Nat := []
theorem innerScan617_8 : scan edge (search edge 3) 4 [4583] 4583 innerTail617_8 best = best := rfl

def innerTail617_7 : List Nat := 7577 :: innerTail617_8
def innerCandidates617_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates617_7_checked : candidates edge 4 4583 7577 innerTail617_8 best.bound = innerCandidates617_7 := by
  unfold candidates
  have row : edge 7577 = SearchData.row7577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan617_7 : scan edge (search edge 3) 4 [4583] 4583 innerTail617_7 best = best := by
  rw [innerTail617_7, scan, if_neg (by decide), innerCandidates617_7_checked]
  change scan edge (search edge 3) 4 [4583] 4583 innerTail617_8
    (if 3 ≤ innerCandidates617_7.length then search edge 3 [4583, 7577] innerCandidates617_7 12160 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan617_8

def innerTail617_6 : List Nat := 7523 :: innerTail617_7
def innerCandidates617_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates617_6_checked : candidates edge 4 4583 7523 innerTail617_7 best.bound = innerCandidates617_6 := by
  unfold candidates
  have row : edge 7523 = SearchData.row7523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan617_6 : scan edge (search edge 3) 4 [4583] 4583 innerTail617_6 best = best := by
  rw [innerTail617_6, scan, if_neg (by decide), innerCandidates617_6_checked]
  change scan edge (search edge 3) 4 [4583] 4583 innerTail617_7
    (if 3 ≤ innerCandidates617_6.length then search edge 3 [4583, 7523] innerCandidates617_6 12106 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan617_7

def innerTail617_5 : List Nat := 5651 :: innerTail617_6
def innerCandidates617_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates617_5_checked : candidates edge 4 4583 5651 innerTail617_6 best.bound = innerCandidates617_5 := by
  unfold candidates
  have row : edge 5651 = SearchData.row5651.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan617_5 : scan edge (search edge 3) 4 [4583] 4583 innerTail617_5 best = best := by
  rw [innerTail617_5, scan, if_neg (by decide), innerCandidates617_5_checked]
  change scan edge (search edge 3) 4 [4583] 4583 innerTail617_6
    (if 3 ≤ innerCandidates617_5.length then search edge 3 [4583, 5651] innerCandidates617_5 10234 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan617_6

def innerTail617_4 : List Nat := 5039 :: innerTail617_5
def innerCandidates617_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates617_4_checked : candidates edge 4 4583 5039 innerTail617_5 best.bound = innerCandidates617_4 := by
  unfold candidates
  have row : edge 5039 = SearchData.row5039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan617_4 : scan edge (search edge 3) 4 [4583] 4583 innerTail617_4 best = best := by
  rw [innerTail617_4, scan, if_neg (by decide), innerCandidates617_4_checked]
  change scan edge (search edge 3) 4 [4583] 4583 innerTail617_5
    (if 3 ≤ innerCandidates617_4.length then search edge 3 [4583, 5039] innerCandidates617_4 9622 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan617_5

def innerTail617_3 : List Nat := 5003 :: innerTail617_4
def innerCandidates617_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates617_3_checked : candidates edge 4 4583 5003 innerTail617_4 best.bound = innerCandidates617_3 := by
  unfold candidates
  have row : edge 5003 = SearchData.row5003.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan617_3 : scan edge (search edge 3) 4 [4583] 4583 innerTail617_3 best = best := by
  rw [innerTail617_3, scan, if_neg (by decide), innerCandidates617_3_checked]
  change scan edge (search edge 3) 4 [4583] 4583 innerTail617_4
    (if 3 ≤ innerCandidates617_3.length then search edge 3 [4583, 5003] innerCandidates617_3 9586 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan617_4

def innerTail617_2 : List Nat := 4871 :: innerTail617_3
def innerCandidates617_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates617_2_checked : candidates edge 4 4583 4871 innerTail617_3 best.bound = innerCandidates617_2 := by
  unfold candidates
  have row : edge 4871 = SearchData.row4871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan617_2 : scan edge (search edge 3) 4 [4583] 4583 innerTail617_2 best = best := by
  rw [innerTail617_2, scan, if_neg (by decide), innerCandidates617_2_checked]
  change scan edge (search edge 3) 4 [4583] 4583 innerTail617_3
    (if 3 ≤ innerCandidates617_2.length then search edge 3 [4583, 4871] innerCandidates617_2 9454 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan617_3

def innerTail617_1 : List Nat := 4787 :: innerTail617_2
def innerCandidates617_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates617_1_checked : candidates edge 4 4583 4787 innerTail617_2 best.bound = innerCandidates617_1 := by
  unfold candidates
  have row : edge 4787 = SearchData.row4787.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan617_1 : scan edge (search edge 3) 4 [4583] 4583 innerTail617_1 best = best := by
  rw [innerTail617_1, scan, if_neg (by decide), innerCandidates617_1_checked]
  change scan edge (search edge 3) 4 [4583] 4583 innerTail617_2
    (if 3 ≤ innerCandidates617_1.length then search edge 3 [4583, 4787] innerCandidates617_1 9370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan617_2

def innerTail617_0 : List Nat := 4637 :: innerTail617_1
def innerCandidates617_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates617_0_checked : candidates edge 4 4583 4637 innerTail617_1 best.bound = innerCandidates617_0 := by
  unfold candidates
  have row : edge 4637 = SearchData.row4637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan617_0 : scan edge (search edge 3) 4 [4583] 4583 innerTail617_0 best = best := by
  rw [innerTail617_0, scan, if_neg (by decide), innerCandidates617_0_checked]
  change scan edge (search edge 3) 4 [4583] 4583 innerTail617_1
    (if 3 ≤ innerCandidates617_0.length then search edge 3 [4583, 4637] innerCandidates617_0 9220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan617_1

theorem rootBranch617 : search edge 4 [4583] rootCandidates617 4583 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4583] 4583 innerTail617_0 best = best
  exact innerScan617_0

def rootCandidates616 : List Nat := [5323, 5557, 7039, 7417, 7789, 7993]
set_option maxRecDepth 4096 in
theorem rootCandidates616_checked : candidates edge 5 0 4567 tail617 best.bound = rootCandidates616 := by
  unfold candidates
  have row : edge 4567 = SearchData.row4567.contains := by funext q; rfl
  rw [row]
  decide

def innerTail616_6 : List Nat := []
theorem innerScan616_6 : scan edge (search edge 3) 4 [4567] 4567 innerTail616_6 best = best := rfl

def innerTail616_5 : List Nat := 7993 :: innerTail616_6
def innerCandidates616_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates616_5_checked : candidates edge 4 4567 7993 innerTail616_6 best.bound = innerCandidates616_5 := by
  unfold candidates
  have row : edge 7993 = SearchData.row7993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan616_5 : scan edge (search edge 3) 4 [4567] 4567 innerTail616_5 best = best := by
  rw [innerTail616_5, scan, if_neg (by decide), innerCandidates616_5_checked]
  change scan edge (search edge 3) 4 [4567] 4567 innerTail616_6
    (if 3 ≤ innerCandidates616_5.length then search edge 3 [4567, 7993] innerCandidates616_5 12560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan616_6

def innerTail616_4 : List Nat := 7789 :: innerTail616_5
def innerCandidates616_4 : List Nat := [7993]
set_option maxRecDepth 4096 in
theorem innerCandidates616_4_checked : candidates edge 4 4567 7789 innerTail616_5 best.bound = innerCandidates616_4 := by
  unfold candidates
  have row : edge 7789 = SearchData.row7789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan616_4 : scan edge (search edge 3) 4 [4567] 4567 innerTail616_4 best = best := by
  rw [innerTail616_4, scan, if_neg (by decide), innerCandidates616_4_checked]
  change scan edge (search edge 3) 4 [4567] 4567 innerTail616_5
    (if 3 ≤ innerCandidates616_4.length then search edge 3 [4567, 7789] innerCandidates616_4 12356 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan616_5

def innerTail616_3 : List Nat := 7417 :: innerTail616_4
def innerCandidates616_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates616_3_checked : candidates edge 4 4567 7417 innerTail616_4 best.bound = innerCandidates616_3 := by
  unfold candidates
  have row : edge 7417 = SearchData.row7417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan616_3 : scan edge (search edge 3) 4 [4567] 4567 innerTail616_3 best = best := by
  rw [innerTail616_3, scan, if_neg (by decide), innerCandidates616_3_checked]
  change scan edge (search edge 3) 4 [4567] 4567 innerTail616_4
    (if 3 ≤ innerCandidates616_3.length then search edge 3 [4567, 7417] innerCandidates616_3 11984 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan616_4

def innerTail616_2 : List Nat := 7039 :: innerTail616_3
def innerCandidates616_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates616_2_checked : candidates edge 4 4567 7039 innerTail616_3 best.bound = innerCandidates616_2 := by
  unfold candidates
  have row : edge 7039 = SearchData.row7039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan616_2 : scan edge (search edge 3) 4 [4567] 4567 innerTail616_2 best = best := by
  rw [innerTail616_2, scan, if_neg (by decide), innerCandidates616_2_checked]
  change scan edge (search edge 3) 4 [4567] 4567 innerTail616_3
    (if 3 ≤ innerCandidates616_2.length then search edge 3 [4567, 7039] innerCandidates616_2 11606 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan616_3

def innerTail616_1 : List Nat := 5557 :: innerTail616_2
def innerCandidates616_1 : List Nat := [7789]
set_option maxRecDepth 4096 in
theorem innerCandidates616_1_checked : candidates edge 4 4567 5557 innerTail616_2 best.bound = innerCandidates616_1 := by
  unfold candidates
  have row : edge 5557 = SearchData.row5557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan616_1 : scan edge (search edge 3) 4 [4567] 4567 innerTail616_1 best = best := by
  rw [innerTail616_1, scan, if_neg (by decide), innerCandidates616_1_checked]
  change scan edge (search edge 3) 4 [4567] 4567 innerTail616_2
    (if 3 ≤ innerCandidates616_1.length then search edge 3 [4567, 5557] innerCandidates616_1 10124 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan616_2

def innerTail616_0 : List Nat := 5323 :: innerTail616_1
def innerCandidates616_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates616_0_checked : candidates edge 4 4567 5323 innerTail616_1 best.bound = innerCandidates616_0 := by
  unfold candidates
  have row : edge 5323 = SearchData.row5323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan616_0 : scan edge (search edge 3) 4 [4567] 4567 innerTail616_0 best = best := by
  rw [innerTail616_0, scan, if_neg (by decide), innerCandidates616_0_checked]
  change scan edge (search edge 3) 4 [4567] 4567 innerTail616_1
    (if 3 ≤ innerCandidates616_0.length then search edge 3 [4567, 5323] innerCandidates616_0 9890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan616_1

theorem rootBranch616 : search edge 4 [4567] rootCandidates616 4567 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4567] 4567 innerTail616_0 best = best
  exact innerScan616_0

def rootCandidates615 : List Nat := [4783, 4909, 5419, 5527, 6073, 6379, 7723, 8017, 8101]
set_option maxRecDepth 4096 in
theorem rootCandidates615_checked : candidates edge 5 0 4561 tail616 best.bound = rootCandidates615 := by
  unfold candidates
  have row : edge 4561 = SearchData.row4561.contains := by funext q; rfl
  rw [row]
  decide

def innerTail615_9 : List Nat := []
theorem innerScan615_9 : scan edge (search edge 3) 4 [4561] 4561 innerTail615_9 best = best := rfl

def innerTail615_8 : List Nat := 8101 :: innerTail615_9
def innerCandidates615_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates615_8_checked : candidates edge 4 4561 8101 innerTail615_9 best.bound = innerCandidates615_8 := by
  unfold candidates
  have row : edge 8101 = SearchData.row8101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan615_8 : scan edge (search edge 3) 4 [4561] 4561 innerTail615_8 best = best := by
  rw [innerTail615_8, scan, if_neg (by decide), innerCandidates615_8_checked]
  change scan edge (search edge 3) 4 [4561] 4561 innerTail615_9
    (if 3 ≤ innerCandidates615_8.length then search edge 3 [4561, 8101] innerCandidates615_8 12662 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan615_9

def innerTail615_7 : List Nat := 8017 :: innerTail615_8
def innerCandidates615_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates615_7_checked : candidates edge 4 4561 8017 innerTail615_8 best.bound = innerCandidates615_7 := by
  unfold candidates
  have row : edge 8017 = SearchData.row8017.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan615_7 : scan edge (search edge 3) 4 [4561] 4561 innerTail615_7 best = best := by
  rw [innerTail615_7, scan, if_neg (by decide), innerCandidates615_7_checked]
  change scan edge (search edge 3) 4 [4561] 4561 innerTail615_8
    (if 3 ≤ innerCandidates615_7.length then search edge 3 [4561, 8017] innerCandidates615_7 12578 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan615_8

def innerTail615_6 : List Nat := 7723 :: innerTail615_7
def innerCandidates615_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates615_6_checked : candidates edge 4 4561 7723 innerTail615_7 best.bound = innerCandidates615_6 := by
  unfold candidates
  have row : edge 7723 = SearchData.row7723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan615_6 : scan edge (search edge 3) 4 [4561] 4561 innerTail615_6 best = best := by
  rw [innerTail615_6, scan, if_neg (by decide), innerCandidates615_6_checked]
  change scan edge (search edge 3) 4 [4561] 4561 innerTail615_7
    (if 3 ≤ innerCandidates615_6.length then search edge 3 [4561, 7723] innerCandidates615_6 12284 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan615_7

def innerTail615_5 : List Nat := 6379 :: innerTail615_6
def innerCandidates615_5 : List Nat := [8017]
set_option maxRecDepth 4096 in
theorem innerCandidates615_5_checked : candidates edge 4 4561 6379 innerTail615_6 best.bound = innerCandidates615_5 := by
  unfold candidates
  have row : edge 6379 = SearchData.row6379.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan615_5 : scan edge (search edge 3) 4 [4561] 4561 innerTail615_5 best = best := by
  rw [innerTail615_5, scan, if_neg (by decide), innerCandidates615_5_checked]
  change scan edge (search edge 3) 4 [4561] 4561 innerTail615_6
    (if 3 ≤ innerCandidates615_5.length then search edge 3 [4561, 6379] innerCandidates615_5 10940 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan615_6

def innerTail615_4 : List Nat := 6073 :: innerTail615_5
def innerCandidates615_4 : List Nat := [7723]
set_option maxRecDepth 4096 in
theorem innerCandidates615_4_checked : candidates edge 4 4561 6073 innerTail615_5 best.bound = innerCandidates615_4 := by
  unfold candidates
  have row : edge 6073 = SearchData.row6073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan615_4 : scan edge (search edge 3) 4 [4561] 4561 innerTail615_4 best = best := by
  rw [innerTail615_4, scan, if_neg (by decide), innerCandidates615_4_checked]
  change scan edge (search edge 3) 4 [4561] 4561 innerTail615_5
    (if 3 ≤ innerCandidates615_4.length then search edge 3 [4561, 6073] innerCandidates615_4 10634 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan615_5

def innerTail615_3 : List Nat := 5527 :: innerTail615_4
def innerCandidates615_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates615_3_checked : candidates edge 4 4561 5527 innerTail615_4 best.bound = innerCandidates615_3 := by
  unfold candidates
  have row : edge 5527 = SearchData.row5527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan615_3 : scan edge (search edge 3) 4 [4561] 4561 innerTail615_3 best = best := by
  rw [innerTail615_3, scan, if_neg (by decide), innerCandidates615_3_checked]
  change scan edge (search edge 3) 4 [4561] 4561 innerTail615_4
    (if 3 ≤ innerCandidates615_3.length then search edge 3 [4561, 5527] innerCandidates615_3 10088 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan615_4

def innerTail615_2 : List Nat := 5419 :: innerTail615_3
def innerCandidates615_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates615_2_checked : candidates edge 4 4561 5419 innerTail615_3 best.bound = innerCandidates615_2 := by
  unfold candidates
  have row : edge 5419 = SearchData.row5419.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan615_2 : scan edge (search edge 3) 4 [4561] 4561 innerTail615_2 best = best := by
  rw [innerTail615_2, scan, if_neg (by decide), innerCandidates615_2_checked]
  change scan edge (search edge 3) 4 [4561] 4561 innerTail615_3
    (if 3 ≤ innerCandidates615_2.length then search edge 3 [4561, 5419] innerCandidates615_2 9980 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan615_3

def innerTail615_1 : List Nat := 4909 :: innerTail615_2
def innerCandidates615_1 : List Nat := [7723]
set_option maxRecDepth 4096 in
theorem innerCandidates615_1_checked : candidates edge 4 4561 4909 innerTail615_2 best.bound = innerCandidates615_1 := by
  unfold candidates
  have row : edge 4909 = SearchData.row4909.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan615_1 : scan edge (search edge 3) 4 [4561] 4561 innerTail615_1 best = best := by
  rw [innerTail615_1, scan, if_neg (by decide), innerCandidates615_1_checked]
  change scan edge (search edge 3) 4 [4561] 4561 innerTail615_2
    (if 3 ≤ innerCandidates615_1.length then search edge 3 [4561, 4909] innerCandidates615_1 9470 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan615_2

def innerTail615_0 : List Nat := 4783 :: innerTail615_1
def innerCandidates615_0 : List Nat := [6073]
set_option maxRecDepth 4096 in
theorem innerCandidates615_0_checked : candidates edge 4 4561 4783 innerTail615_1 best.bound = innerCandidates615_0 := by
  unfold candidates
  have row : edge 4783 = SearchData.row4783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan615_0 : scan edge (search edge 3) 4 [4561] 4561 innerTail615_0 best = best := by
  rw [innerTail615_0, scan, if_neg (by decide), innerCandidates615_0_checked]
  change scan edge (search edge 3) 4 [4561] 4561 innerTail615_1
    (if 3 ≤ innerCandidates615_0.length then search edge 3 [4561, 4783] innerCandidates615_0 9344 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan615_1

theorem rootBranch615 : search edge 4 [4561] rootCandidates615 4561 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4561] 4561 innerTail615_0 best = best
  exact innerScan615_0

def rootCandidates614 : List Nat := [5563, 5839, 5851, 6133, 6151, 6343, 6361, 6571, 7717, 8287]
set_option maxRecDepth 4096 in
theorem rootCandidates614_checked : candidates edge 5 0 4549 tail615 best.bound = rootCandidates614 := by
  unfold candidates
  have row : edge 4549 = SearchData.row4549.contains := by funext q; rfl
  rw [row]
  decide

def innerTail614_10 : List Nat := []
theorem innerScan614_10 : scan edge (search edge 3) 4 [4549] 4549 innerTail614_10 best = best := rfl

def innerTail614_9 : List Nat := 8287 :: innerTail614_10
def innerCandidates614_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates614_9_checked : candidates edge 4 4549 8287 innerTail614_10 best.bound = innerCandidates614_9 := by
  unfold candidates
  have row : edge 8287 = SearchData.row8287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan614_9 : scan edge (search edge 3) 4 [4549] 4549 innerTail614_9 best = best := by
  rw [innerTail614_9, scan, if_neg (by decide), innerCandidates614_9_checked]
  change scan edge (search edge 3) 4 [4549] 4549 innerTail614_10
    (if 3 ≤ innerCandidates614_9.length then search edge 3 [4549, 8287] innerCandidates614_9 12836 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan614_10

def innerTail614_8 : List Nat := 7717 :: innerTail614_9
def innerCandidates614_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates614_8_checked : candidates edge 4 4549 7717 innerTail614_9 best.bound = innerCandidates614_8 := by
  unfold candidates
  have row : edge 7717 = SearchData.row7717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan614_8 : scan edge (search edge 3) 4 [4549] 4549 innerTail614_8 best = best := by
  rw [innerTail614_8, scan, if_neg (by decide), innerCandidates614_8_checked]
  change scan edge (search edge 3) 4 [4549] 4549 innerTail614_9
    (if 3 ≤ innerCandidates614_8.length then search edge 3 [4549, 7717] innerCandidates614_8 12266 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan614_9

def innerTail614_7 : List Nat := 6571 :: innerTail614_8
def innerCandidates614_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates614_7_checked : candidates edge 4 4549 6571 innerTail614_8 best.bound = innerCandidates614_7 := by
  unfold candidates
  have row : edge 6571 = SearchData.row6571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan614_7 : scan edge (search edge 3) 4 [4549] 4549 innerTail614_7 best = best := by
  rw [innerTail614_7, scan, if_neg (by decide), innerCandidates614_7_checked]
  change scan edge (search edge 3) 4 [4549] 4549 innerTail614_8
    (if 3 ≤ innerCandidates614_7.length then search edge 3 [4549, 6571] innerCandidates614_7 11120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan614_8

def innerTail614_6 : List Nat := 6361 :: innerTail614_7
def innerCandidates614_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates614_6_checked : candidates edge 4 4549 6361 innerTail614_7 best.bound = innerCandidates614_6 := by
  unfold candidates
  have row : edge 6361 = SearchData.row6361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan614_6 : scan edge (search edge 3) 4 [4549] 4549 innerTail614_6 best = best := by
  rw [innerTail614_6, scan, if_neg (by decide), innerCandidates614_6_checked]
  change scan edge (search edge 3) 4 [4549] 4549 innerTail614_7
    (if 3 ≤ innerCandidates614_6.length then search edge 3 [4549, 6361] innerCandidates614_6 10910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan614_7

def innerTail614_5 : List Nat := 6343 :: innerTail614_6
def innerCandidates614_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates614_5_checked : candidates edge 4 4549 6343 innerTail614_6 best.bound = innerCandidates614_5 := by
  unfold candidates
  have row : edge 6343 = SearchData.row6343.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan614_5 : scan edge (search edge 3) 4 [4549] 4549 innerTail614_5 best = best := by
  rw [innerTail614_5, scan, if_neg (by decide), innerCandidates614_5_checked]
  change scan edge (search edge 3) 4 [4549] 4549 innerTail614_6
    (if 3 ≤ innerCandidates614_5.length then search edge 3 [4549, 6343] innerCandidates614_5 10892 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan614_6

def innerTail614_4 : List Nat := 6151 :: innerTail614_5
def innerCandidates614_4 : List Nat := [6571]
set_option maxRecDepth 4096 in
theorem innerCandidates614_4_checked : candidates edge 4 4549 6151 innerTail614_5 best.bound = innerCandidates614_4 := by
  unfold candidates
  have row : edge 6151 = SearchData.row6151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan614_4 : scan edge (search edge 3) 4 [4549] 4549 innerTail614_4 best = best := by
  rw [innerTail614_4, scan, if_neg (by decide), innerCandidates614_4_checked]
  change scan edge (search edge 3) 4 [4549] 4549 innerTail614_5
    (if 3 ≤ innerCandidates614_4.length then search edge 3 [4549, 6151] innerCandidates614_4 10700 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan614_5

def innerTail614_3 : List Nat := 6133 :: innerTail614_4
def innerCandidates614_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates614_3_checked : candidates edge 4 4549 6133 innerTail614_4 best.bound = innerCandidates614_3 := by
  unfold candidates
  have row : edge 6133 = SearchData.row6133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan614_3 : scan edge (search edge 3) 4 [4549] 4549 innerTail614_3 best = best := by
  rw [innerTail614_3, scan, if_neg (by decide), innerCandidates614_3_checked]
  change scan edge (search edge 3) 4 [4549] 4549 innerTail614_4
    (if 3 ≤ innerCandidates614_3.length then search edge 3 [4549, 6133] innerCandidates614_3 10682 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan614_4

def innerTail614_2 : List Nat := 5851 :: innerTail614_3
def innerCandidates614_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates614_2_checked : candidates edge 4 4549 5851 innerTail614_3 best.bound = innerCandidates614_2 := by
  unfold candidates
  have row : edge 5851 = SearchData.row5851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan614_2 : scan edge (search edge 3) 4 [4549] 4549 innerTail614_2 best = best := by
  rw [innerTail614_2, scan, if_neg (by decide), innerCandidates614_2_checked]
  change scan edge (search edge 3) 4 [4549] 4549 innerTail614_3
    (if 3 ≤ innerCandidates614_2.length then search edge 3 [4549, 5851] innerCandidates614_2 10400 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan614_3

def innerTail614_1 : List Nat := 5839 :: innerTail614_2
def innerCandidates614_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates614_1_checked : candidates edge 4 4549 5839 innerTail614_2 best.bound = innerCandidates614_1 := by
  unfold candidates
  have row : edge 5839 = SearchData.row5839.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan614_1 : scan edge (search edge 3) 4 [4549] 4549 innerTail614_1 best = best := by
  rw [innerTail614_1, scan, if_neg (by decide), innerCandidates614_1_checked]
  change scan edge (search edge 3) 4 [4549] 4549 innerTail614_2
    (if 3 ≤ innerCandidates614_1.length then search edge 3 [4549, 5839] innerCandidates614_1 10388 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan614_2

def innerTail614_0 : List Nat := 5563 :: innerTail614_1
def innerCandidates614_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates614_0_checked : candidates edge 4 4549 5563 innerTail614_1 best.bound = innerCandidates614_0 := by
  unfold candidates
  have row : edge 5563 = SearchData.row5563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan614_0 : scan edge (search edge 3) 4 [4549] 4549 innerTail614_0 best = best := by
  rw [innerTail614_0, scan, if_neg (by decide), innerCandidates614_0_checked]
  change scan edge (search edge 3) 4 [4549] 4549 innerTail614_1
    (if 3 ≤ innerCandidates614_0.length then search edge 3 [4549, 5563] innerCandidates614_0 10112 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan614_1

theorem rootBranch614 : search edge 4 [4549] rootCandidates614 4549 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4549] 4549 innerTail614_0 best = best
  exact innerScan614_0

def rootCandidates613 : List Nat := [4943, 4973, 6311, 6521, 6689, 6791, 6863, 7283, 7547, 7919]
set_option maxRecDepth 4096 in
theorem rootCandidates613_checked : candidates edge 5 0 4547 tail614 best.bound = rootCandidates613 := by
  unfold candidates
  have row : edge 4547 = SearchData.row4547.contains := by funext q; rfl
  rw [row]
  decide

def innerTail613_10 : List Nat := []
theorem innerScan613_10 : scan edge (search edge 3) 4 [4547] 4547 innerTail613_10 best = best := rfl

def innerTail613_9 : List Nat := 7919 :: innerTail613_10
def innerCandidates613_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates613_9_checked : candidates edge 4 4547 7919 innerTail613_10 best.bound = innerCandidates613_9 := by
  unfold candidates
  have row : edge 7919 = SearchData.row7919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan613_9 : scan edge (search edge 3) 4 [4547] 4547 innerTail613_9 best = best := by
  rw [innerTail613_9, scan, if_neg (by decide), innerCandidates613_9_checked]
  change scan edge (search edge 3) 4 [4547] 4547 innerTail613_10
    (if 3 ≤ innerCandidates613_9.length then search edge 3 [4547, 7919] innerCandidates613_9 12466 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan613_10

def innerTail613_8 : List Nat := 7547 :: innerTail613_9
def innerCandidates613_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates613_8_checked : candidates edge 4 4547 7547 innerTail613_9 best.bound = innerCandidates613_8 := by
  unfold candidates
  have row : edge 7547 = SearchData.row7547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan613_8 : scan edge (search edge 3) 4 [4547] 4547 innerTail613_8 best = best := by
  rw [innerTail613_8, scan, if_neg (by decide), innerCandidates613_8_checked]
  change scan edge (search edge 3) 4 [4547] 4547 innerTail613_9
    (if 3 ≤ innerCandidates613_8.length then search edge 3 [4547, 7547] innerCandidates613_8 12094 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan613_9

def innerTail613_7 : List Nat := 7283 :: innerTail613_8
def innerCandidates613_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates613_7_checked : candidates edge 4 4547 7283 innerTail613_8 best.bound = innerCandidates613_7 := by
  unfold candidates
  have row : edge 7283 = SearchData.row7283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan613_7 : scan edge (search edge 3) 4 [4547] 4547 innerTail613_7 best = best := by
  rw [innerTail613_7, scan, if_neg (by decide), innerCandidates613_7_checked]
  change scan edge (search edge 3) 4 [4547] 4547 innerTail613_8
    (if 3 ≤ innerCandidates613_7.length then search edge 3 [4547, 7283] innerCandidates613_7 11830 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan613_8

def innerTail613_6 : List Nat := 6863 :: innerTail613_7
def innerCandidates613_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates613_6_checked : candidates edge 4 4547 6863 innerTail613_7 best.bound = innerCandidates613_6 := by
  unfold candidates
  have row : edge 6863 = SearchData.row6863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan613_6 : scan edge (search edge 3) 4 [4547] 4547 innerTail613_6 best = best := by
  rw [innerTail613_6, scan, if_neg (by decide), innerCandidates613_6_checked]
  change scan edge (search edge 3) 4 [4547] 4547 innerTail613_7
    (if 3 ≤ innerCandidates613_6.length then search edge 3 [4547, 6863] innerCandidates613_6 11410 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan613_7

def innerTail613_5 : List Nat := 6791 :: innerTail613_6
def innerCandidates613_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates613_5_checked : candidates edge 4 4547 6791 innerTail613_6 best.bound = innerCandidates613_5 := by
  unfold candidates
  have row : edge 6791 = SearchData.row6791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan613_5 : scan edge (search edge 3) 4 [4547] 4547 innerTail613_5 best = best := by
  rw [innerTail613_5, scan, if_neg (by decide), innerCandidates613_5_checked]
  change scan edge (search edge 3) 4 [4547] 4547 innerTail613_6
    (if 3 ≤ innerCandidates613_5.length then search edge 3 [4547, 6791] innerCandidates613_5 11338 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan613_6

def innerTail613_4 : List Nat := 6689 :: innerTail613_5
def innerCandidates613_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates613_4_checked : candidates edge 4 4547 6689 innerTail613_5 best.bound = innerCandidates613_4 := by
  unfold candidates
  have row : edge 6689 = SearchData.row6689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan613_4 : scan edge (search edge 3) 4 [4547] 4547 innerTail613_4 best = best := by
  rw [innerTail613_4, scan, if_neg (by decide), innerCandidates613_4_checked]
  change scan edge (search edge 3) 4 [4547] 4547 innerTail613_5
    (if 3 ≤ innerCandidates613_4.length then search edge 3 [4547, 6689] innerCandidates613_4 11236 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan613_5

def innerTail613_3 : List Nat := 6521 :: innerTail613_4
def innerCandidates613_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates613_3_checked : candidates edge 4 4547 6521 innerTail613_4 best.bound = innerCandidates613_3 := by
  unfold candidates
  have row : edge 6521 = SearchData.row6521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan613_3 : scan edge (search edge 3) 4 [4547] 4547 innerTail613_3 best = best := by
  rw [innerTail613_3, scan, if_neg (by decide), innerCandidates613_3_checked]
  change scan edge (search edge 3) 4 [4547] 4547 innerTail613_4
    (if 3 ≤ innerCandidates613_3.length then search edge 3 [4547, 6521] innerCandidates613_3 11068 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan613_4

def innerTail613_2 : List Nat := 6311 :: innerTail613_3
def innerCandidates613_2 : List Nat := [6863]
set_option maxRecDepth 4096 in
theorem innerCandidates613_2_checked : candidates edge 4 4547 6311 innerTail613_3 best.bound = innerCandidates613_2 := by
  unfold candidates
  have row : edge 6311 = SearchData.row6311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan613_2 : scan edge (search edge 3) 4 [4547] 4547 innerTail613_2 best = best := by
  rw [innerTail613_2, scan, if_neg (by decide), innerCandidates613_2_checked]
  change scan edge (search edge 3) 4 [4547] 4547 innerTail613_3
    (if 3 ≤ innerCandidates613_2.length then search edge 3 [4547, 6311] innerCandidates613_2 10858 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan613_3

def innerTail613_1 : List Nat := 4973 :: innerTail613_2
def innerCandidates613_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates613_1_checked : candidates edge 4 4547 4973 innerTail613_2 best.bound = innerCandidates613_1 := by
  unfold candidates
  have row : edge 4973 = SearchData.row4973.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan613_1 : scan edge (search edge 3) 4 [4547] 4547 innerTail613_1 best = best := by
  rw [innerTail613_1, scan, if_neg (by decide), innerCandidates613_1_checked]
  change scan edge (search edge 3) 4 [4547] 4547 innerTail613_2
    (if 3 ≤ innerCandidates613_1.length then search edge 3 [4547, 4973] innerCandidates613_1 9520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan613_2

def innerTail613_0 : List Nat := 4943 :: innerTail613_1
def innerCandidates613_0 : List Nat := [7919]
set_option maxRecDepth 4096 in
theorem innerCandidates613_0_checked : candidates edge 4 4547 4943 innerTail613_1 best.bound = innerCandidates613_0 := by
  unfold candidates
  have row : edge 4943 = SearchData.row4943.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan613_0 : scan edge (search edge 3) 4 [4547] 4547 innerTail613_0 best = best := by
  rw [innerTail613_0, scan, if_neg (by decide), innerCandidates613_0_checked]
  change scan edge (search edge 3) 4 [4547] 4547 innerTail613_1
    (if 3 ≤ innerCandidates613_0.length then search edge 3 [4547, 4943] innerCandidates613_0 9490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan613_1

theorem rootBranch613 : search edge 4 [4547] rootCandidates613 4547 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4547] 4547 innerTail613_0 best = best
  exact innerScan613_0

end ProjectEulerSolutions.PrimePairUpper.Search
