import Mathlib.Data.List.Basic

/-! Select the first later stop in a listed route, wrapping to its starting stop. -/
namespace ProjectEulerSolutions.CyclicStops

def nextStop (stops : List Int) (start position : Int) : Int :=
  match stops with
  | [] => start
  | stop :: rest => if position < stop then stop else nextStop rest start position

theorem nextStop_eq_find (stops : List Int) (start position : Int) :
    nextStop stops start position = (stops.find? fun stop => decide (position < stop)).getD start := by
  induction stops with
  | nil => rfl
  | cons stop rest ih =>
      by_cases later : position < stop <;> simp [nextStop, later, ih]

theorem nextStop_mem (stops : List Int) (start position : Int) :
    nextStop stops start position = start ∨ nextStop stops start position ∈ stops := by
  induction stops with
  | nil => exact Or.inl rfl
  | cons stop rest ih =>
      unfold nextStop
      split
      · exact Or.inr (List.mem_cons_self ..)
      · exact ih.imp_right (List.mem_cons_of_mem stop)

end ProjectEulerSolutions.CyclicStops
