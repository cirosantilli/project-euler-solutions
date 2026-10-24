import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates927 : List Nat := [7507, 7951, 8053, 8179, 8377]
set_option maxRecDepth 4096 in
theorem rootCandidates927_checked : candidates edge 5 0 7297 tail928 best.bound = rootCandidates927 := by
  unfold candidates
  have row : edge 7297 = SearchData.row7297.contains := by funext q; rfl
  rw [row]
  decide

def innerTail927_5 : List Nat := []
theorem innerScan927_5 : scan edge (search edge 3) 4 [7297] 7297 innerTail927_5 best = best := rfl

def innerTail927_4 : List Nat := 8377 :: innerTail927_5
def innerCandidates927_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates927_4_checked : candidates edge 4 7297 8377 innerTail927_5 best.bound = innerCandidates927_4 := by
  unfold candidates
  have row : edge 8377 = SearchData.row8377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan927_4 : scan edge (search edge 3) 4 [7297] 7297 innerTail927_4 best = best := by
  rw [innerTail927_4, scan, if_neg (by decide), innerCandidates927_4_checked]
  change scan edge (search edge 3) 4 [7297] 7297 innerTail927_5
    (if 3 ≤ innerCandidates927_4.length then search edge 3 [7297, 8377] innerCandidates927_4 15674 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan927_5

def innerTail927_3 : List Nat := 8179 :: innerTail927_4
def innerCandidates927_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates927_3_checked : candidates edge 4 7297 8179 innerTail927_4 best.bound = innerCandidates927_3 := by
  unfold candidates
  have row : edge 8179 = SearchData.row8179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan927_3 : scan edge (search edge 3) 4 [7297] 7297 innerTail927_3 best = best := by
  rw [innerTail927_3, scan, if_neg (by decide), innerCandidates927_3_checked]
  change scan edge (search edge 3) 4 [7297] 7297 innerTail927_4
    (if 3 ≤ innerCandidates927_3.length then search edge 3 [7297, 8179] innerCandidates927_3 15476 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan927_4

def innerTail927_2 : List Nat := 8053 :: innerTail927_3
def innerCandidates927_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates927_2_checked : candidates edge 4 7297 8053 innerTail927_3 best.bound = innerCandidates927_2 := by
  unfold candidates
  have row : edge 8053 = SearchData.row8053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan927_2 : scan edge (search edge 3) 4 [7297] 7297 innerTail927_2 best = best := by
  rw [innerTail927_2, scan, if_neg (by decide), innerCandidates927_2_checked]
  change scan edge (search edge 3) 4 [7297] 7297 innerTail927_3
    (if 3 ≤ innerCandidates927_2.length then search edge 3 [7297, 8053] innerCandidates927_2 15350 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan927_3

def innerTail927_1 : List Nat := 7951 :: innerTail927_2
def innerCandidates927_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates927_1_checked : candidates edge 4 7297 7951 innerTail927_2 best.bound = innerCandidates927_1 := by
  unfold candidates
  have row : edge 7951 = SearchData.row7951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan927_1 : scan edge (search edge 3) 4 [7297] 7297 innerTail927_1 best = best := by
  rw [innerTail927_1, scan, if_neg (by decide), innerCandidates927_1_checked]
  change scan edge (search edge 3) 4 [7297] 7297 innerTail927_2
    (if 3 ≤ innerCandidates927_1.length then search edge 3 [7297, 7951] innerCandidates927_1 15248 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan927_2

def innerTail927_0 : List Nat := 7507 :: innerTail927_1
def innerCandidates927_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates927_0_checked : candidates edge 4 7297 7507 innerTail927_1 best.bound = innerCandidates927_0 := by
  unfold candidates
  have row : edge 7507 = SearchData.row7507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan927_0 : scan edge (search edge 3) 4 [7297] 7297 innerTail927_0 best = best := by
  rw [innerTail927_0, scan, if_neg (by decide), innerCandidates927_0_checked]
  change scan edge (search edge 3) 4 [7297] 7297 innerTail927_1
    (if 3 ≤ innerCandidates927_0.length then search edge 3 [7297, 7507] innerCandidates927_0 14804 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan927_1

theorem rootBranch927 : search edge 4 [7297] rootCandidates927 7297 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [7297] 7297 innerTail927_0 best = best
  exact innerScan927_0

def rootCandidates926 : List Nat := [7577, 7649, 7901, 7949, 8147, 8243]
set_option maxRecDepth 4096 in
theorem rootCandidates926_checked : candidates edge 5 0 7283 tail927 best.bound = rootCandidates926 := by
  unfold candidates
  have row : edge 7283 = SearchData.row7283.contains := by funext q; rfl
  rw [row]
  decide

def innerTail926_6 : List Nat := []
theorem innerScan926_6 : scan edge (search edge 3) 4 [7283] 7283 innerTail926_6 best = best := rfl

def innerTail926_5 : List Nat := 8243 :: innerTail926_6
def innerCandidates926_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates926_5_checked : candidates edge 4 7283 8243 innerTail926_6 best.bound = innerCandidates926_5 := by
  unfold candidates
  have row : edge 8243 = SearchData.row8243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan926_5 : scan edge (search edge 3) 4 [7283] 7283 innerTail926_5 best = best := by
  rw [innerTail926_5, scan, if_neg (by decide), innerCandidates926_5_checked]
  change scan edge (search edge 3) 4 [7283] 7283 innerTail926_6
    (if 3 ≤ innerCandidates926_5.length then search edge 3 [7283, 8243] innerCandidates926_5 15526 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan926_6

def innerTail926_4 : List Nat := 8147 :: innerTail926_5
def innerCandidates926_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates926_4_checked : candidates edge 4 7283 8147 innerTail926_5 best.bound = innerCandidates926_4 := by
  unfold candidates
  have row : edge 8147 = SearchData.row8147.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan926_4 : scan edge (search edge 3) 4 [7283] 7283 innerTail926_4 best = best := by
  rw [innerTail926_4, scan, if_neg (by decide), innerCandidates926_4_checked]
  change scan edge (search edge 3) 4 [7283] 7283 innerTail926_5
    (if 3 ≤ innerCandidates926_4.length then search edge 3 [7283, 8147] innerCandidates926_4 15430 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan926_5

def innerTail926_3 : List Nat := 7949 :: innerTail926_4
def innerCandidates926_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates926_3_checked : candidates edge 4 7283 7949 innerTail926_4 best.bound = innerCandidates926_3 := by
  unfold candidates
  have row : edge 7949 = SearchData.row7949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan926_3 : scan edge (search edge 3) 4 [7283] 7283 innerTail926_3 best = best := by
  rw [innerTail926_3, scan, if_neg (by decide), innerCandidates926_3_checked]
  change scan edge (search edge 3) 4 [7283] 7283 innerTail926_4
    (if 3 ≤ innerCandidates926_3.length then search edge 3 [7283, 7949] innerCandidates926_3 15232 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan926_4

def innerTail926_2 : List Nat := 7901 :: innerTail926_3
def innerCandidates926_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates926_2_checked : candidates edge 4 7283 7901 innerTail926_3 best.bound = innerCandidates926_2 := by
  unfold candidates
  have row : edge 7901 = SearchData.row7901.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan926_2 : scan edge (search edge 3) 4 [7283] 7283 innerTail926_2 best = best := by
  rw [innerTail926_2, scan, if_neg (by decide), innerCandidates926_2_checked]
  change scan edge (search edge 3) 4 [7283] 7283 innerTail926_3
    (if 3 ≤ innerCandidates926_2.length then search edge 3 [7283, 7901] innerCandidates926_2 15184 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan926_3

def innerTail926_1 : List Nat := 7649 :: innerTail926_2
def innerCandidates926_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates926_1_checked : candidates edge 4 7283 7649 innerTail926_2 best.bound = innerCandidates926_1 := by
  unfold candidates
  have row : edge 7649 = SearchData.row7649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan926_1 : scan edge (search edge 3) 4 [7283] 7283 innerTail926_1 best = best := by
  rw [innerTail926_1, scan, if_neg (by decide), innerCandidates926_1_checked]
  change scan edge (search edge 3) 4 [7283] 7283 innerTail926_2
    (if 3 ≤ innerCandidates926_1.length then search edge 3 [7283, 7649] innerCandidates926_1 14932 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan926_2

def innerTail926_0 : List Nat := 7577 :: innerTail926_1
def innerCandidates926_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates926_0_checked : candidates edge 4 7283 7577 innerTail926_1 best.bound = innerCandidates926_0 := by
  unfold candidates
  have row : edge 7577 = SearchData.row7577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan926_0 : scan edge (search edge 3) 4 [7283] 7283 innerTail926_0 best = best := by
  rw [innerTail926_0, scan, if_neg (by decide), innerCandidates926_0_checked]
  change scan edge (search edge 3) 4 [7283] 7283 innerTail926_1
    (if 3 ≤ innerCandidates926_0.length then search edge 3 [7283, 7577] innerCandidates926_0 14860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan926_1

theorem rootBranch926 : search edge 4 [7283] rootCandidates926 7283 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [7283] 7283 innerTail926_0 best = best
  exact innerScan926_0

def rootCandidates925 : List Nat := [7499, 8009, 8243]
set_option maxRecDepth 4096 in
theorem rootCandidates925_checked : candidates edge 5 0 7253 tail926 best.bound = rootCandidates925 := by
  unfold candidates
  have row : edge 7253 = SearchData.row7253.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates924 : List Nat := [7331, 7541, 8111]
set_option maxRecDepth 4096 in
theorem rootCandidates924_checked : candidates edge 5 0 7247 tail925 best.bound = rootCandidates924 := by
  unfold candidates
  have row : edge 7247 = SearchData.row7247.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates923 : List Nat := [8209]
set_option maxRecDepth 4096 in
theorem rootCandidates923_checked : candidates edge 5 0 7243 tail924 best.bound = rootCandidates923 := by
  unfold candidates
  have row : edge 7243 = SearchData.row7243.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates922 : List Nat := [7507, 7561, 7699, 7759]
set_option maxRecDepth 4096 in
theorem rootCandidates922_checked : candidates edge 5 0 7237 tail923 best.bound = rootCandidates922 := by
  unfold candidates
  have row : edge 7237 = SearchData.row7237.contains := by funext q; rfl
  rw [row]
  decide

def innerTail922_4 : List Nat := []
theorem innerScan922_4 : scan edge (search edge 3) 4 [7237] 7237 innerTail922_4 best = best := rfl

def innerTail922_3 : List Nat := 7759 :: innerTail922_4
def innerCandidates922_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates922_3_checked : candidates edge 4 7237 7759 innerTail922_4 best.bound = innerCandidates922_3 := by
  unfold candidates
  have row : edge 7759 = SearchData.row7759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan922_3 : scan edge (search edge 3) 4 [7237] 7237 innerTail922_3 best = best := by
  rw [innerTail922_3, scan, if_neg (by decide), innerCandidates922_3_checked]
  change scan edge (search edge 3) 4 [7237] 7237 innerTail922_4
    (if 3 ≤ innerCandidates922_3.length then search edge 3 [7237, 7759] innerCandidates922_3 14996 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan922_4

def innerTail922_2 : List Nat := 7699 :: innerTail922_3
def innerCandidates922_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates922_2_checked : candidates edge 4 7237 7699 innerTail922_3 best.bound = innerCandidates922_2 := by
  unfold candidates
  have row : edge 7699 = SearchData.row7699.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan922_2 : scan edge (search edge 3) 4 [7237] 7237 innerTail922_2 best = best := by
  rw [innerTail922_2, scan, if_neg (by decide), innerCandidates922_2_checked]
  change scan edge (search edge 3) 4 [7237] 7237 innerTail922_3
    (if 3 ≤ innerCandidates922_2.length then search edge 3 [7237, 7699] innerCandidates922_2 14936 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan922_3

def innerTail922_1 : List Nat := 7561 :: innerTail922_2
def innerCandidates922_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates922_1_checked : candidates edge 4 7237 7561 innerTail922_2 best.bound = innerCandidates922_1 := by
  unfold candidates
  have row : edge 7561 = SearchData.row7561.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan922_1 : scan edge (search edge 3) 4 [7237] 7237 innerTail922_1 best = best := by
  rw [innerTail922_1, scan, if_neg (by decide), innerCandidates922_1_checked]
  change scan edge (search edge 3) 4 [7237] 7237 innerTail922_2
    (if 3 ≤ innerCandidates922_1.length then search edge 3 [7237, 7561] innerCandidates922_1 14798 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan922_2

def innerTail922_0 : List Nat := 7507 :: innerTail922_1
def innerCandidates922_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates922_0_checked : candidates edge 4 7237 7507 innerTail922_1 best.bound = innerCandidates922_0 := by
  unfold candidates
  have row : edge 7507 = SearchData.row7507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan922_0 : scan edge (search edge 3) 4 [7237] 7237 innerTail922_0 best = best := by
  rw [innerTail922_0, scan, if_neg (by decide), innerCandidates922_0_checked]
  change scan edge (search edge 3) 4 [7237] 7237 innerTail922_1
    (if 3 ≤ innerCandidates922_0.length then search edge 3 [7237, 7507] innerCandidates922_0 14744 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan922_1

theorem rootBranch922 : search edge 4 [7237] rootCandidates922 7237 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [7237] 7237 innerTail922_0 best = best
  exact innerScan922_0

def rootCandidates921 : List Nat := [7307, 7541, 7691, 7829, 7937, 8147]
set_option maxRecDepth 4096 in
theorem rootCandidates921_checked : candidates edge 5 0 7229 tail922 best.bound = rootCandidates921 := by
  unfold candidates
  have row : edge 7229 = SearchData.row7229.contains := by funext q; rfl
  rw [row]
  decide

def innerTail921_6 : List Nat := []
theorem innerScan921_6 : scan edge (search edge 3) 4 [7229] 7229 innerTail921_6 best = best := rfl

def innerTail921_5 : List Nat := 8147 :: innerTail921_6
def innerCandidates921_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates921_5_checked : candidates edge 4 7229 8147 innerTail921_6 best.bound = innerCandidates921_5 := by
  unfold candidates
  have row : edge 8147 = SearchData.row8147.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan921_5 : scan edge (search edge 3) 4 [7229] 7229 innerTail921_5 best = best := by
  rw [innerTail921_5, scan, if_neg (by decide), innerCandidates921_5_checked]
  change scan edge (search edge 3) 4 [7229] 7229 innerTail921_6
    (if 3 ≤ innerCandidates921_5.length then search edge 3 [7229, 8147] innerCandidates921_5 15376 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan921_6

def innerTail921_4 : List Nat := 7937 :: innerTail921_5
def innerCandidates921_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates921_4_checked : candidates edge 4 7229 7937 innerTail921_5 best.bound = innerCandidates921_4 := by
  unfold candidates
  have row : edge 7937 = SearchData.row7937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan921_4 : scan edge (search edge 3) 4 [7229] 7229 innerTail921_4 best = best := by
  rw [innerTail921_4, scan, if_neg (by decide), innerCandidates921_4_checked]
  change scan edge (search edge 3) 4 [7229] 7229 innerTail921_5
    (if 3 ≤ innerCandidates921_4.length then search edge 3 [7229, 7937] innerCandidates921_4 15166 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan921_5

def innerTail921_3 : List Nat := 7829 :: innerTail921_4
def innerCandidates921_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates921_3_checked : candidates edge 4 7229 7829 innerTail921_4 best.bound = innerCandidates921_3 := by
  unfold candidates
  have row : edge 7829 = SearchData.row7829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan921_3 : scan edge (search edge 3) 4 [7229] 7229 innerTail921_3 best = best := by
  rw [innerTail921_3, scan, if_neg (by decide), innerCandidates921_3_checked]
  change scan edge (search edge 3) 4 [7229] 7229 innerTail921_4
    (if 3 ≤ innerCandidates921_3.length then search edge 3 [7229, 7829] innerCandidates921_3 15058 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan921_4

def innerTail921_2 : List Nat := 7691 :: innerTail921_3
def innerCandidates921_2 : List Nat := [8147]
set_option maxRecDepth 4096 in
theorem innerCandidates921_2_checked : candidates edge 4 7229 7691 innerTail921_3 best.bound = innerCandidates921_2 := by
  unfold candidates
  have row : edge 7691 = SearchData.row7691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan921_2 : scan edge (search edge 3) 4 [7229] 7229 innerTail921_2 best = best := by
  rw [innerTail921_2, scan, if_neg (by decide), innerCandidates921_2_checked]
  change scan edge (search edge 3) 4 [7229] 7229 innerTail921_3
    (if 3 ≤ innerCandidates921_2.length then search edge 3 [7229, 7691] innerCandidates921_2 14920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan921_3

def innerTail921_1 : List Nat := 7541 :: innerTail921_2
def innerCandidates921_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates921_1_checked : candidates edge 4 7229 7541 innerTail921_2 best.bound = innerCandidates921_1 := by
  unfold candidates
  have row : edge 7541 = SearchData.row7541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan921_1 : scan edge (search edge 3) 4 [7229] 7229 innerTail921_1 best = best := by
  rw [innerTail921_1, scan, if_neg (by decide), innerCandidates921_1_checked]
  change scan edge (search edge 3) 4 [7229] 7229 innerTail921_2
    (if 3 ≤ innerCandidates921_1.length then search edge 3 [7229, 7541] innerCandidates921_1 14770 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan921_2

def innerTail921_0 : List Nat := 7307 :: innerTail921_1
def innerCandidates921_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates921_0_checked : candidates edge 4 7229 7307 innerTail921_1 best.bound = innerCandidates921_0 := by
  unfold candidates
  have row : edge 7307 = SearchData.row7307.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan921_0 : scan edge (search edge 3) 4 [7229] 7229 innerTail921_0 best = best := by
  rw [innerTail921_0, scan, if_neg (by decide), innerCandidates921_0_checked]
  change scan edge (search edge 3) 4 [7229] 7229 innerTail921_1
    (if 3 ≤ innerCandidates921_0.length then search edge 3 [7229, 7307] innerCandidates921_0 14536 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan921_1

theorem rootBranch921 : search edge 4 [7229] rootCandidates921 7229 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [7229] 7229 innerTail921_0 best = best
  exact innerScan921_0

def rootCandidates920 : List Nat := [7537, 7669, 8209, 8317, 8377]
set_option maxRecDepth 4096 in
theorem rootCandidates920_checked : candidates edge 5 0 7219 tail921 best.bound = rootCandidates920 := by
  unfold candidates
  have row : edge 7219 = SearchData.row7219.contains := by funext q; rfl
  rw [row]
  decide

def innerTail920_5 : List Nat := []
theorem innerScan920_5 : scan edge (search edge 3) 4 [7219] 7219 innerTail920_5 best = best := rfl

def innerTail920_4 : List Nat := 8377 :: innerTail920_5
def innerCandidates920_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates920_4_checked : candidates edge 4 7219 8377 innerTail920_5 best.bound = innerCandidates920_4 := by
  unfold candidates
  have row : edge 8377 = SearchData.row8377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan920_4 : scan edge (search edge 3) 4 [7219] 7219 innerTail920_4 best = best := by
  rw [innerTail920_4, scan, if_neg (by decide), innerCandidates920_4_checked]
  change scan edge (search edge 3) 4 [7219] 7219 innerTail920_5
    (if 3 ≤ innerCandidates920_4.length then search edge 3 [7219, 8377] innerCandidates920_4 15596 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan920_5

def innerTail920_3 : List Nat := 8317 :: innerTail920_4
def innerCandidates920_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates920_3_checked : candidates edge 4 7219 8317 innerTail920_4 best.bound = innerCandidates920_3 := by
  unfold candidates
  have row : edge 8317 = SearchData.row8317.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan920_3 : scan edge (search edge 3) 4 [7219] 7219 innerTail920_3 best = best := by
  rw [innerTail920_3, scan, if_neg (by decide), innerCandidates920_3_checked]
  change scan edge (search edge 3) 4 [7219] 7219 innerTail920_4
    (if 3 ≤ innerCandidates920_3.length then search edge 3 [7219, 8317] innerCandidates920_3 15536 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan920_4

def innerTail920_2 : List Nat := 8209 :: innerTail920_3
def innerCandidates920_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates920_2_checked : candidates edge 4 7219 8209 innerTail920_3 best.bound = innerCandidates920_2 := by
  unfold candidates
  have row : edge 8209 = SearchData.row8209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan920_2 : scan edge (search edge 3) 4 [7219] 7219 innerTail920_2 best = best := by
  rw [innerTail920_2, scan, if_neg (by decide), innerCandidates920_2_checked]
  change scan edge (search edge 3) 4 [7219] 7219 innerTail920_3
    (if 3 ≤ innerCandidates920_2.length then search edge 3 [7219, 8209] innerCandidates920_2 15428 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan920_3

def innerTail920_1 : List Nat := 7669 :: innerTail920_2
def innerCandidates920_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates920_1_checked : candidates edge 4 7219 7669 innerTail920_2 best.bound = innerCandidates920_1 := by
  unfold candidates
  have row : edge 7669 = SearchData.row7669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan920_1 : scan edge (search edge 3) 4 [7219] 7219 innerTail920_1 best = best := by
  rw [innerTail920_1, scan, if_neg (by decide), innerCandidates920_1_checked]
  change scan edge (search edge 3) 4 [7219] 7219 innerTail920_2
    (if 3 ≤ innerCandidates920_1.length then search edge 3 [7219, 7669] innerCandidates920_1 14888 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan920_2

def innerTail920_0 : List Nat := 7537 :: innerTail920_1
def innerCandidates920_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates920_0_checked : candidates edge 4 7219 7537 innerTail920_1 best.bound = innerCandidates920_0 := by
  unfold candidates
  have row : edge 7537 = SearchData.row7537.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan920_0 : scan edge (search edge 3) 4 [7219] 7219 innerTail920_0 best = best := by
  rw [innerTail920_0, scan, if_neg (by decide), innerCandidates920_0_checked]
  change scan edge (search edge 3) 4 [7219] 7219 innerTail920_1
    (if 3 ≤ innerCandidates920_0.length then search edge 3 [7219, 7537] innerCandidates920_0 14756 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan920_1

theorem rootBranch920 : search edge 4 [7219] rootCandidates920 7219 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [7219] 7219 innerTail920_0 best = best
  exact innerScan920_0

end ProjectEulerSolutions.PrimePairUpper.Search
