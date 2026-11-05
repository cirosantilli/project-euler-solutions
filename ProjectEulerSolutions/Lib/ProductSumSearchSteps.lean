import ProjectEulerSolutions.Lib.ProductSumTable

/-! Loop equations for the total preorder trace of the factor search. -/
namespace ProjectEulerSolutions.ProductSumNumbers
set_option Elab.async false

def branch (maxSize limit product summ terms factor : Nat) : List (Nat × Nat) :=
  let size := terms + 1 + product * factor - (summ + factor)
  if size ≤ maxSize then
    (size, product * factor) :: search maxSize limit factor (product * factor) (summ + factor) (terms + 1)
  else []

theorem search_eq (maxSize limit start product summ terms : Nat)
    (positive : 0 < product) (lower : 2 ≤ start) :
    search maxSize limit start product summ terms =
      (nextFactors limit start product).flatMap (branch maxSize limit product summ terms) := by
  rw [search, dif_pos ⟨positive, lower⟩]
  rw [List.flatMap_subtype (g := branch maxSize limit product summ terms) (by intros; rfl),
    List.unattach_attach]

theorem search_done (maxSize limit start product summ terms : Nat)
    (positive : 0 < product) (lower : 2 ≤ start) (done : limit / product < start) :
    search maxSize limit start product summ terms = [] := by
  rw [search_eq maxSize limit start product summ terms positive lower]
  simp [nextFactors, show limit / product + 1 - start = 0 by omega]

theorem nextFactors_step (limit start product : Nat) (active : start ≤ limit / product) :
    nextFactors limit start product = start :: nextFactors limit (start + 1) product := by
  unfold nextFactors
  rw [show limit / product + 1 - start = (limit / product + 1 - (start + 1)) + 1 by omega]
  simp [List.range'_succ]

theorem search_step (maxSize limit start product summ terms : Nat)
    (positive : 0 < product) (lower : 2 ≤ start) (active : start ≤ limit / product) :
    search maxSize limit start product summ terms =
      branch maxSize limit product summ terms start ++
        search maxSize limit (start + 1) product summ terms := by
  rw [search_eq maxSize limit start product summ terms positive lower,
    nextFactors_step limit start product active, List.flatMap_cons,
    search_eq maxSize limit (start + 1) product summ terms positive (by omega)]

theorem accumulate_step (maxSize limit start product summ terms : Nat) (best : List Nat)
    (positive : 0 < product) (lower : 2 ≤ start) (active : start ≤ limit / product) :
    accumulate (search maxSize limit start product summ terms) best =
      accumulate (search maxSize limit (start + 1) product summ terms)
        (accumulate (branch maxSize limit product summ terms start) best) := by
  rw [search_step maxSize limit start product summ terms positive lower active, accumulate_append]

end ProjectEulerSolutions.ProductSumNumbers
