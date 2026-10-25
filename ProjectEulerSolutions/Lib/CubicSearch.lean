import ProjectEulerSolutions.Lib.CubicGroups

/-! The digit-width search, including finite witnesses for its unbounded loop. -/
namespace ProjectEulerSolutions.CubicSearch
open CubicGroups CubicPermutations
set_option Elab.async false

def cube (n : Nat) : Nat := n * n * n
def width (n : Nat) : Nat := (decimal (cube n)).length

structure State where
  next : Nat
  currentWidth : Nat
  groups : List Group

def advance (state : State) (currentWidth : Nat) (groups : List Group) : State :=
  ⟨state.next + 1, currentWidth, record (cube state.next) groups⟩

def step (target : Int) (state : State) : Sum Int State :=
  if state.currentWidth < width state.next then
    match matching target state.groups with
    | [] => .inr (advance state (width state.next) [])
    | first :: rest => .inl (rest.foldl min first)
  else .inr (advance state state.currentWidth state.groups)

/-- A witness describes a finite successful run; no bound on the target or search is assumed. -/
inductive Returns (target : Int) : State → Int → Prop
  | done {state : State} {value : Int} : step target state = .inl value → Returns target state value
  | next {state following : State} {value : Int} :
      step target state = .inr following → Returns target following value → Returns target state value

def start : State := ⟨1, 1, []⟩
def ready : State := ⟨2, 1, [fresh 1]⟩

theorem step_start (target : Int) : step target start = .inr ready := by
  simp [step, start, show width 1 = 1 from by decide, advance, ready, cube]

theorem Returns.ready_of_start {target value : Int} (returned : Returns target start value) :
    Returns target ready value := by
  cases returned with
  | done checked => simp [step_start] at checked
  | next checked rest =>
      rw [step_start] at checked
      cases checked
      exact rest

end ProjectEulerSolutions.CubicSearch
