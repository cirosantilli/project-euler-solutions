import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates911 : List Nat := [7411]
set_option maxRecDepth 4096 in
theorem rootCandidates911_checked : candidates edge 5 0 7129 tail912 best.bound = rootCandidates911 := by
  unfold candidates
  have row : edge 7129 = SearchData.row7129.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates910 : List Nat := [7757, 7793]
set_option maxRecDepth 4096 in
theorem rootCandidates910_checked : candidates edge 5 0 7127 tail911 best.bound = rootCandidates910 := by
  unfold candidates
  have row : edge 7127 = SearchData.row7127.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates909 : List Nat := [8297]
set_option maxRecDepth 4096 in
theorem rootCandidates909_checked : candidates edge 5 0 7121 tail910 best.bound = rootCandidates909 := by
  unfold candidates
  have row : edge 7121 = SearchData.row7121.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates908 : List Nat := [7127, 7841, 7907]
set_option maxRecDepth 4096 in
theorem rootCandidates908_checked : candidates edge 5 0 7109 tail909 best.bound = rootCandidates908 := by
  unfold candidates
  have row : edge 7109 = SearchData.row7109.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates907 : List Nat := [7283, 8273]
set_option maxRecDepth 4096 in
theorem rootCandidates907_checked : candidates edge 5 0 7103 tail908 best.bound = rootCandidates907 := by
  unfold candidates
  have row : edge 7103 = SearchData.row7103.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates906 : List Nat := [7229, 8111, 8363]
set_option maxRecDepth 4096 in
theorem rootCandidates906_checked : candidates edge 5 0 7079 tail907 best.bound = rootCandidates906 := by
  unfold candidates
  have row : edge 7079 = SearchData.row7079.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates905 : List Nat := [7369, 7741, 8161]
set_option maxRecDepth 4096 in
theorem rootCandidates905_checked : candidates edge 5 0 7069 tail906 best.bound = rootCandidates905 := by
  unfold candidates
  have row : edge 7069 = SearchData.row7069.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates904 : List Nat := [8353]
set_option maxRecDepth 4096 in
theorem rootCandidates904_checked : candidates edge 5 0 7057 tail905 best.bound = rootCandidates904 := by
  unfold candidates
  have row : edge 7057 = SearchData.row7057.contains := by funext q; rfl
  rw [row]
  decide

end ProjectEulerSolutions.PrimePairUpper.Search
