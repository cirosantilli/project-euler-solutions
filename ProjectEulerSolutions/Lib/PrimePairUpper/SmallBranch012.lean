import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates951 : List Nat := [7877]
set_option maxRecDepth 4096 in
theorem rootCandidates951_checked : candidates edge 5 0 7529 tail952 best.bound = rootCandidates951 := by
  unfold candidates
  have row : edge 7529 = SearchData.row7529.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates950 : List Nat := [7541, 7727]
set_option maxRecDepth 4096 in
theorem rootCandidates950_checked : candidates edge 5 0 7523 tail951 best.bound = rootCandidates950 := by
  unfold candidates
  have row : edge 7523 = SearchData.row7523.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates949 : List Nat := [7727, 7829, 8387]
set_option maxRecDepth 4096 in
theorem rootCandidates949_checked : candidates edge 5 0 7517 tail950 best.bound = rootCandidates949 := by
  unfold candidates
  have row : edge 7517 = SearchData.row7517.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates948 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates948_checked : candidates edge 5 0 7507 tail949 best.bound = rootCandidates948 := by
  unfold candidates
  have row : edge 7507 = SearchData.row7507.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates947 : List Nat := [7757, 7841, 8369]
set_option maxRecDepth 4096 in
theorem rootCandidates947_checked : candidates edge 5 0 7499 tail948 best.bound = rootCandidates947 := by
  unfold candidates
  have row : edge 7499 = SearchData.row7499.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates946 : List Nat := [7549]
set_option maxRecDepth 4096 in
theorem rootCandidates946_checked : candidates edge 5 0 7489 tail947 best.bound = rootCandidates946 := by
  unfold candidates
  have row : edge 7489 = SearchData.row7489.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates945 : List Nat := [7883]
set_option maxRecDepth 4096 in
theorem rootCandidates945_checked : candidates edge 5 0 7487 tail946 best.bound = rootCandidates945 := by
  unfold candidates
  have row : edge 7487 = SearchData.row7487.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates944 : List Nat := [7541, 7607]
set_option maxRecDepth 4096 in
theorem rootCandidates944_checked : candidates edge 5 0 7481 tail945 best.bound = rootCandidates944 := by
  unfold candidates
  have row : edge 7481 = SearchData.row7481.contains := by funext q; rfl
  rw [row]
  decide

end ProjectEulerSolutions.PrimePairUpper.Search
