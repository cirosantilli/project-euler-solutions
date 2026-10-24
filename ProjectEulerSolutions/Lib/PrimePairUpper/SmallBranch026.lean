import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates839 : List Nat := [6563, 6983, 7253, 7583, 8237]
set_option maxRecDepth 4096 in
theorem rootCandidates839_checked : candidates edge 5 0 6491 tail840 best.bound = rootCandidates839 := by
  unfold candidates
  have row : edge 6491 = SearchData.row6491.contains := by funext q; rfl
  rw [row]
  decide

def innerTail839_5 : List Nat := []
theorem innerScan839_5 : scan edge (search edge 3) 4 [6491] 6491 innerTail839_5 best = best := rfl

def innerTail839_4 : List Nat := 8237 :: innerTail839_5
def innerCandidates839_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates839_4_checked : candidates edge 4 6491 8237 innerTail839_5 best.bound = innerCandidates839_4 := by
  unfold candidates
  have row : edge 8237 = SearchData.row8237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan839_4 : scan edge (search edge 3) 4 [6491] 6491 innerTail839_4 best = best := by
  rw [innerTail839_4, scan, if_neg (by decide), innerCandidates839_4_checked]
  change scan edge (search edge 3) 4 [6491] 6491 innerTail839_5
    (if 3 ≤ innerCandidates839_4.length then search edge 3 [6491, 8237] innerCandidates839_4 14728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan839_5

def innerTail839_3 : List Nat := 7583 :: innerTail839_4
def innerCandidates839_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates839_3_checked : candidates edge 4 6491 7583 innerTail839_4 best.bound = innerCandidates839_3 := by
  unfold candidates
  have row : edge 7583 = SearchData.row7583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan839_3 : scan edge (search edge 3) 4 [6491] 6491 innerTail839_3 best = best := by
  rw [innerTail839_3, scan, if_neg (by decide), innerCandidates839_3_checked]
  change scan edge (search edge 3) 4 [6491] 6491 innerTail839_4
    (if 3 ≤ innerCandidates839_3.length then search edge 3 [6491, 7583] innerCandidates839_3 14074 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan839_4

def innerTail839_2 : List Nat := 7253 :: innerTail839_3
def innerCandidates839_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates839_2_checked : candidates edge 4 6491 7253 innerTail839_3 best.bound = innerCandidates839_2 := by
  unfold candidates
  have row : edge 7253 = SearchData.row7253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan839_2 : scan edge (search edge 3) 4 [6491] 6491 innerTail839_2 best = best := by
  rw [innerTail839_2, scan, if_neg (by decide), innerCandidates839_2_checked]
  change scan edge (search edge 3) 4 [6491] 6491 innerTail839_3
    (if 3 ≤ innerCandidates839_2.length then search edge 3 [6491, 7253] innerCandidates839_2 13744 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan839_3

def innerTail839_1 : List Nat := 6983 :: innerTail839_2
def innerCandidates839_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates839_1_checked : candidates edge 4 6491 6983 innerTail839_2 best.bound = innerCandidates839_1 := by
  unfold candidates
  have row : edge 6983 = SearchData.row6983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan839_1 : scan edge (search edge 3) 4 [6491] 6491 innerTail839_1 best = best := by
  rw [innerTail839_1, scan, if_neg (by decide), innerCandidates839_1_checked]
  change scan edge (search edge 3) 4 [6491] 6491 innerTail839_2
    (if 3 ≤ innerCandidates839_1.length then search edge 3 [6491, 6983] innerCandidates839_1 13474 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan839_2

def innerTail839_0 : List Nat := 6563 :: innerTail839_1
def innerCandidates839_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates839_0_checked : candidates edge 4 6491 6563 innerTail839_1 best.bound = innerCandidates839_0 := by
  unfold candidates
  have row : edge 6563 = SearchData.row6563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan839_0 : scan edge (search edge 3) 4 [6491] 6491 innerTail839_0 best = best := by
  rw [innerTail839_0, scan, if_neg (by decide), innerCandidates839_0_checked]
  change scan edge (search edge 3) 4 [6491] 6491 innerTail839_1
    (if 3 ≤ innerCandidates839_0.length then search edge 3 [6491, 6563] innerCandidates839_0 13054 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan839_1

theorem rootBranch839 : search edge 4 [6491] rootCandidates839 6491 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6491] 6491 innerTail839_0 best = best
  exact innerScan839_0

def rootCandidates838 : List Nat := [6793, 7603, 8287]
set_option maxRecDepth 4096 in
theorem rootCandidates838_checked : candidates edge 5 0 6481 tail839 best.bound = rootCandidates838 := by
  unfold candidates
  have row : edge 6481 = SearchData.row6481.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates837 : List Nat := [6701, 7643]
set_option maxRecDepth 4096 in
theorem rootCandidates837_checked : candidates edge 5 0 6473 tail838 best.bound = rootCandidates837 := by
  unfold candidates
  have row : edge 6473 = SearchData.row6473.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates836 : List Nat := [6481, 6673, 6703, 6997, 7219, 7681, 7879, 8233]
set_option maxRecDepth 4096 in
theorem rootCandidates836_checked : candidates edge 5 0 6469 tail837 best.bound = rootCandidates836 := by
  unfold candidates
  have row : edge 6469 = SearchData.row6469.contains := by funext q; rfl
  rw [row]
  decide

def innerTail836_8 : List Nat := []
theorem innerScan836_8 : scan edge (search edge 3) 4 [6469] 6469 innerTail836_8 best = best := rfl

def innerTail836_7 : List Nat := 8233 :: innerTail836_8
def innerCandidates836_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates836_7_checked : candidates edge 4 6469 8233 innerTail836_8 best.bound = innerCandidates836_7 := by
  unfold candidates
  have row : edge 8233 = SearchData.row8233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan836_7 : scan edge (search edge 3) 4 [6469] 6469 innerTail836_7 best = best := by
  rw [innerTail836_7, scan, if_neg (by decide), innerCandidates836_7_checked]
  change scan edge (search edge 3) 4 [6469] 6469 innerTail836_8
    (if 3 ≤ innerCandidates836_7.length then search edge 3 [6469, 8233] innerCandidates836_7 14702 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan836_8

def innerTail836_6 : List Nat := 7879 :: innerTail836_7
def innerCandidates836_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates836_6_checked : candidates edge 4 6469 7879 innerTail836_7 best.bound = innerCandidates836_6 := by
  unfold candidates
  have row : edge 7879 = SearchData.row7879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan836_6 : scan edge (search edge 3) 4 [6469] 6469 innerTail836_6 best = best := by
  rw [innerTail836_6, scan, if_neg (by decide), innerCandidates836_6_checked]
  change scan edge (search edge 3) 4 [6469] 6469 innerTail836_7
    (if 3 ≤ innerCandidates836_6.length then search edge 3 [6469, 7879] innerCandidates836_6 14348 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan836_7

def innerTail836_5 : List Nat := 7681 :: innerTail836_6
def innerCandidates836_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates836_5_checked : candidates edge 4 6469 7681 innerTail836_6 best.bound = innerCandidates836_5 := by
  unfold candidates
  have row : edge 7681 = SearchData.row7681.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan836_5 : scan edge (search edge 3) 4 [6469] 6469 innerTail836_5 best = best := by
  rw [innerTail836_5, scan, if_neg (by decide), innerCandidates836_5_checked]
  change scan edge (search edge 3) 4 [6469] 6469 innerTail836_6
    (if 3 ≤ innerCandidates836_5.length then search edge 3 [6469, 7681] innerCandidates836_5 14150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan836_6

def innerTail836_4 : List Nat := 7219 :: innerTail836_5
def innerCandidates836_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates836_4_checked : candidates edge 4 6469 7219 innerTail836_5 best.bound = innerCandidates836_4 := by
  unfold candidates
  have row : edge 7219 = SearchData.row7219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan836_4 : scan edge (search edge 3) 4 [6469] 6469 innerTail836_4 best = best := by
  rw [innerTail836_4, scan, if_neg (by decide), innerCandidates836_4_checked]
  change scan edge (search edge 3) 4 [6469] 6469 innerTail836_5
    (if 3 ≤ innerCandidates836_4.length then search edge 3 [6469, 7219] innerCandidates836_4 13688 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan836_5

def innerTail836_3 : List Nat := 6997 :: innerTail836_4
def innerCandidates836_3 : List Nat := [7879]
set_option maxRecDepth 4096 in
theorem innerCandidates836_3_checked : candidates edge 4 6469 6997 innerTail836_4 best.bound = innerCandidates836_3 := by
  unfold candidates
  have row : edge 6997 = SearchData.row6997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan836_3 : scan edge (search edge 3) 4 [6469] 6469 innerTail836_3 best = best := by
  rw [innerTail836_3, scan, if_neg (by decide), innerCandidates836_3_checked]
  change scan edge (search edge 3) 4 [6469] 6469 innerTail836_4
    (if 3 ≤ innerCandidates836_3.length then search edge 3 [6469, 6997] innerCandidates836_3 13466 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan836_4

def innerTail836_2 : List Nat := 6703 :: innerTail836_3
def innerCandidates836_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates836_2_checked : candidates edge 4 6469 6703 innerTail836_3 best.bound = innerCandidates836_2 := by
  unfold candidates
  have row : edge 6703 = SearchData.row6703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan836_2 : scan edge (search edge 3) 4 [6469] 6469 innerTail836_2 best = best := by
  rw [innerTail836_2, scan, if_neg (by decide), innerCandidates836_2_checked]
  change scan edge (search edge 3) 4 [6469] 6469 innerTail836_3
    (if 3 ≤ innerCandidates836_2.length then search edge 3 [6469, 6703] innerCandidates836_2 13172 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan836_3

def innerTail836_1 : List Nat := 6673 :: innerTail836_2
def innerCandidates836_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates836_1_checked : candidates edge 4 6469 6673 innerTail836_2 best.bound = innerCandidates836_1 := by
  unfold candidates
  have row : edge 6673 = SearchData.row6673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan836_1 : scan edge (search edge 3) 4 [6469] 6469 innerTail836_1 best = best := by
  rw [innerTail836_1, scan, if_neg (by decide), innerCandidates836_1_checked]
  change scan edge (search edge 3) 4 [6469] 6469 innerTail836_2
    (if 3 ≤ innerCandidates836_1.length then search edge 3 [6469, 6673] innerCandidates836_1 13142 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan836_2

def innerTail836_0 : List Nat := 6481 :: innerTail836_1
def innerCandidates836_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates836_0_checked : candidates edge 4 6469 6481 innerTail836_1 best.bound = innerCandidates836_0 := by
  unfold candidates
  have row : edge 6481 = SearchData.row6481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan836_0 : scan edge (search edge 3) 4 [6469] 6469 innerTail836_0 best = best := by
  rw [innerTail836_0, scan, if_neg (by decide), innerCandidates836_0_checked]
  change scan edge (search edge 3) 4 [6469] 6469 innerTail836_1
    (if 3 ≤ innerCandidates836_0.length then search edge 3 [6469, 6481] innerCandidates836_0 12950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan836_1

theorem rootBranch836 : search edge 4 [6469] rootCandidates836 6469 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6469] 6469 innerTail836_0 best = best
  exact innerScan836_0

def rootCandidates835 : List Nat := [7213, 7309, 7561, 7753, 8191]
set_option maxRecDepth 4096 in
theorem rootCandidates835_checked : candidates edge 5 0 6451 tail836 best.bound = rootCandidates835 := by
  unfold candidates
  have row : edge 6451 = SearchData.row6451.contains := by funext q; rfl
  rw [row]
  decide

def innerTail835_5 : List Nat := []
theorem innerScan835_5 : scan edge (search edge 3) 4 [6451] 6451 innerTail835_5 best = best := rfl

def innerTail835_4 : List Nat := 8191 :: innerTail835_5
def innerCandidates835_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates835_4_checked : candidates edge 4 6451 8191 innerTail835_5 best.bound = innerCandidates835_4 := by
  unfold candidates
  have row : edge 8191 = SearchData.row8191.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan835_4 : scan edge (search edge 3) 4 [6451] 6451 innerTail835_4 best = best := by
  rw [innerTail835_4, scan, if_neg (by decide), innerCandidates835_4_checked]
  change scan edge (search edge 3) 4 [6451] 6451 innerTail835_5
    (if 3 ≤ innerCandidates835_4.length then search edge 3 [6451, 8191] innerCandidates835_4 14642 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan835_5

def innerTail835_3 : List Nat := 7753 :: innerTail835_4
def innerCandidates835_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates835_3_checked : candidates edge 4 6451 7753 innerTail835_4 best.bound = innerCandidates835_3 := by
  unfold candidates
  have row : edge 7753 = SearchData.row7753.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan835_3 : scan edge (search edge 3) 4 [6451] 6451 innerTail835_3 best = best := by
  rw [innerTail835_3, scan, if_neg (by decide), innerCandidates835_3_checked]
  change scan edge (search edge 3) 4 [6451] 6451 innerTail835_4
    (if 3 ≤ innerCandidates835_3.length then search edge 3 [6451, 7753] innerCandidates835_3 14204 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan835_4

def innerTail835_2 : List Nat := 7561 :: innerTail835_3
def innerCandidates835_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates835_2_checked : candidates edge 4 6451 7561 innerTail835_3 best.bound = innerCandidates835_2 := by
  unfold candidates
  have row : edge 7561 = SearchData.row7561.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan835_2 : scan edge (search edge 3) 4 [6451] 6451 innerTail835_2 best = best := by
  rw [innerTail835_2, scan, if_neg (by decide), innerCandidates835_2_checked]
  change scan edge (search edge 3) 4 [6451] 6451 innerTail835_3
    (if 3 ≤ innerCandidates835_2.length then search edge 3 [6451, 7561] innerCandidates835_2 14012 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan835_3

def innerTail835_1 : List Nat := 7309 :: innerTail835_2
def innerCandidates835_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates835_1_checked : candidates edge 4 6451 7309 innerTail835_2 best.bound = innerCandidates835_1 := by
  unfold candidates
  have row : edge 7309 = SearchData.row7309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan835_1 : scan edge (search edge 3) 4 [6451] 6451 innerTail835_1 best = best := by
  rw [innerTail835_1, scan, if_neg (by decide), innerCandidates835_1_checked]
  change scan edge (search edge 3) 4 [6451] 6451 innerTail835_2
    (if 3 ≤ innerCandidates835_1.length then search edge 3 [6451, 7309] innerCandidates835_1 13760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan835_2

def innerTail835_0 : List Nat := 7213 :: innerTail835_1
def innerCandidates835_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates835_0_checked : candidates edge 4 6451 7213 innerTail835_1 best.bound = innerCandidates835_0 := by
  unfold candidates
  have row : edge 7213 = SearchData.row7213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan835_0 : scan edge (search edge 3) 4 [6451] 6451 innerTail835_0 best = best := by
  rw [innerTail835_0, scan, if_neg (by decide), innerCandidates835_0_checked]
  change scan edge (search edge 3) 4 [6451] 6451 innerTail835_1
    (if 3 ≤ innerCandidates835_0.length then search edge 3 [6451, 7213] innerCandidates835_0 13664 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan835_1

theorem rootBranch835 : search edge 4 [6451] rootCandidates835 6451 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6451] 6451 innerTail835_0 best = best
  exact innerScan835_0

def rootCandidates834 : List Nat := [7523, 8171, 8219]
set_option maxRecDepth 4096 in
theorem rootCandidates834_checked : candidates edge 5 0 6449 tail835 best.bound = rootCandidates834 := by
  unfold candidates
  have row : edge 6449 = SearchData.row6449.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates833 : List Nat := [6997, 7069, 8101, 8233]
set_option maxRecDepth 4096 in
theorem rootCandidates833_checked : candidates edge 5 0 6427 tail834 best.bound = rootCandidates833 := by
  unfold candidates
  have row : edge 6427 = SearchData.row6427.contains := by funext q; rfl
  rw [row]
  decide

def innerTail833_4 : List Nat := []
theorem innerScan833_4 : scan edge (search edge 3) 4 [6427] 6427 innerTail833_4 best = best := rfl

def innerTail833_3 : List Nat := 8233 :: innerTail833_4
def innerCandidates833_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates833_3_checked : candidates edge 4 6427 8233 innerTail833_4 best.bound = innerCandidates833_3 := by
  unfold candidates
  have row : edge 8233 = SearchData.row8233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan833_3 : scan edge (search edge 3) 4 [6427] 6427 innerTail833_3 best = best := by
  rw [innerTail833_3, scan, if_neg (by decide), innerCandidates833_3_checked]
  change scan edge (search edge 3) 4 [6427] 6427 innerTail833_4
    (if 3 ≤ innerCandidates833_3.length then search edge 3 [6427, 8233] innerCandidates833_3 14660 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan833_4

def innerTail833_2 : List Nat := 8101 :: innerTail833_3
def innerCandidates833_2 : List Nat := [8233]
set_option maxRecDepth 4096 in
theorem innerCandidates833_2_checked : candidates edge 4 6427 8101 innerTail833_3 best.bound = innerCandidates833_2 := by
  unfold candidates
  have row : edge 8101 = SearchData.row8101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan833_2 : scan edge (search edge 3) 4 [6427] 6427 innerTail833_2 best = best := by
  rw [innerTail833_2, scan, if_neg (by decide), innerCandidates833_2_checked]
  change scan edge (search edge 3) 4 [6427] 6427 innerTail833_3
    (if 3 ≤ innerCandidates833_2.length then search edge 3 [6427, 8101] innerCandidates833_2 14528 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan833_3

def innerTail833_1 : List Nat := 7069 :: innerTail833_2
def innerCandidates833_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates833_1_checked : candidates edge 4 6427 7069 innerTail833_2 best.bound = innerCandidates833_1 := by
  unfold candidates
  have row : edge 7069 = SearchData.row7069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan833_1 : scan edge (search edge 3) 4 [6427] 6427 innerTail833_1 best = best := by
  rw [innerTail833_1, scan, if_neg (by decide), innerCandidates833_1_checked]
  change scan edge (search edge 3) 4 [6427] 6427 innerTail833_2
    (if 3 ≤ innerCandidates833_1.length then search edge 3 [6427, 7069] innerCandidates833_1 13496 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan833_2

def innerTail833_0 : List Nat := 6997 :: innerTail833_1
def innerCandidates833_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates833_0_checked : candidates edge 4 6427 6997 innerTail833_1 best.bound = innerCandidates833_0 := by
  unfold candidates
  have row : edge 6997 = SearchData.row6997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan833_0 : scan edge (search edge 3) 4 [6427] 6427 innerTail833_0 best = best := by
  rw [innerTail833_0, scan, if_neg (by decide), innerCandidates833_0_checked]
  change scan edge (search edge 3) 4 [6427] 6427 innerTail833_1
    (if 3 ≤ innerCandidates833_0.length then search edge 3 [6427, 6997] innerCandidates833_0 13424 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan833_1

theorem rootBranch833 : search edge 4 [6427] rootCandidates833 6427 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6427] 6427 innerTail833_0 best = best
  exact innerScan833_0

def rootCandidates832 : List Nat := [6553, 6949, 8377]
set_option maxRecDepth 4096 in
theorem rootCandidates832_checked : candidates edge 5 0 6421 tail833 best.bound = rootCandidates832 := by
  unfold candidates
  have row : edge 6421 = SearchData.row6421.contains := by funext q; rfl
  rw [row]
  decide

end ProjectEulerSolutions.PrimePairUpper.Search
