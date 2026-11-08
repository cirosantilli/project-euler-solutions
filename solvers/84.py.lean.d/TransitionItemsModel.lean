import solvers.«84.py.lean.d».TransitionOrdinary
import solvers.«84.py.lean.d».LandingWeightLinear
import solvers.«84.py.lean.d».OutcomeTransition

namespace solvers.«84».transition_row
open FormalSW.PyAst ProjectEulerSolutions NaturalDictionary resolve_landing
set_option Elab.async false

def rowWeight (pos doubles move : Nat) (isDouble : Bool) (target : Nat) : Int :=
  ((outcomes.categoryWeights (pos, doubles) move isDouble).map fun entry =>
    if target = Monopoly.encode entry.1.1 entry.1.2 then (entry.2 : Int) else 0).sum

theorem lookup_fold (original : List weights.Entry) (entries : Entries) (doubles count target : Nat) (isDouble : Bool) :
    ((lookup (original.foldl (items.applyEntry doubles count isDouble) entries) target).getD 0 : Int) =
      ((lookup entries target).getD 0 : Int) +
        (original.map fun entry => if target = Monopoly.encode entry.landing.1.toNat (nextDoubles entry.landing.2 isDouble doubles)
          then (entry.weight.toNat : Int) * count else 0).sum := by
  induction original generalizing entries with
  | nil => simp
  | cons entry rest ih =>
      rw [List.foldl_cons, ih]
      simp [items.applyEntry, lookup_add, Int.add_assoc]

theorem landing_sum (dictionary : landings.Dictionary) (heap : Mutable.Heap) (square doubles count target : Nat) (isDouble : Bool)
    (view : dictionary.View heap square) :
    (dictionary.entries.map fun entry =>
      if target = Monopoly.encode entry.landing.1.toNat (nextDoubles entry.landing.2 isDouble doubles)
      then (entry.weight.toNat : Int) * count else 0).sum =
      (count : Int) * ((Monopoly.landingWeights square).map fun entry =>
        if target = Monopoly.encode entry.1.1 (nextDoubles entry.1.2 isDouble doubles) then (entry.2 : Int) else 0).sum := by
  let f := fun landing : Int × Bool =>
    if target = Monopoly.encode landing.1.toNat (nextDoubles landing.2 isDouble doubles) then (1 : Int) else 0
  have rearranged : (dictionary.entries.map fun entry =>
      if target = Monopoly.encode entry.landing.1.toNat (nextDoubles entry.landing.2 isDouble doubles)
      then (entry.weight.toNat : Int) * count else 0) =
      dictionary.entries.map (fun entry => (count : Int) * (entry.weight * f entry.landing)) := by
    apply List.map_congr_left
    intro entry member
    have nonnegative := Int.le_of_lt (view.2.2.2.1 entry member)
    rw [Int.toNat_of_nonneg nonnegative]
    by_cases selected : target = Monopoly.encode entry.landing.1.toNat (nextDoubles entry.landing.2 isDouble doubles) <;>
      simp [f, selected, Int.mul_comm]
  rw [rearranged, List.sum_map_mul_left]
  rw [weights.entries_linear dictionary.entries (modelData square) view.2.2.1 view.2.2.2.2 f]
  congr 1
  simp [modelData, List.map_map, Function.comp_def, f, mul_ite]

/-- The actual dictionary fold has exactly the model's encoded transition contribution. -/
theorem ordinary_lookup (dictionary : landings.Dictionary) (heap : Mutable.Heap)
    (pos move doubles count target : Nat) (isDouble : Bool) (entries : Entries)
    (ordinary : ¬(isDouble = true ∧ doubles = 2)) (view : dictionary.View heap ((pos + move) % 40)) :
    ((lookup (dictionary.entries.foldl (items.applyEntry doubles count isDouble) entries) target).getD 0 : Int) =
      ((lookup entries target).getD 0 : Int) + (count : Int) * rowWeight pos doubles move isDouble target := by
  rw [lookup_fold, landing_sum dictionary heap _ doubles count target isDouble view]
  simp [rowWeight, outcomes.categoryWeights, ordinary, List.map_map, Function.comp_def, nextDoubles]
  exact Or.inl rfl

theorem rowWeight_jail (pos move target : Nat) : rowWeight pos 2 move true target = if target = 30 then 256 else 0 := by
  simp [rowWeight, outcomes.categoryWeights, Monopoly.encode]

end solvers.«84».transition_row
