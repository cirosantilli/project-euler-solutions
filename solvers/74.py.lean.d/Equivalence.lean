import solvers.«74.py.lean.d».Execution
import data.«project-euler-statements».data.py.project_euler_statements.«p74.py»
import lang.py.FormalSW.MutableRelocation

/-! Execute the factorial initialization and compare both original calls in its shared context. -/
namespace solvers.«74».source_equivalence
open FormalSW.PyAst HeapCalls
set_option Elab.async false

abbrev integer := solve.integer
abbrev solveArgs := ((findFunctionIn Solvers.P74Ast.module.body "solve").getD ([], [])).1

def left (supplied : List Mutable.Value) (caller : Mutable.Heap) : Mutable.Call :=
  HeapCalls.call Solvers.P74Ast.module "solve" supplied
    (factorials.heap caller) (factorials.globals caller)

def right (limit target : Int) (caller : Mutable.Heap) : Mutable.Call :=
  HeapCalls.call P74Ast.module "naive" [integer limit, integer target]
    (factorials.heap caller) (factorials.globals caller)

def Initialized (caller : Mutable.Heap) : Prop :=
  BlockContinues Solvers.P74Ast.module (fun _ => none) factorials.code caller
    (factorials.env caller) (factorials.heap caller) [] ∧
  ∀ name, factorials.env caller name = (factorials.globals caller).lookup name

def VerifiedCall (supplied : List Mutable.Value) (limit target : Int)
    (caller : Mutable.Heap) : Prop :=
  Initialized caller ∧ (left supplied caller).Returns ∧ (right limit target caller).Returns ∧
    (left supplied caller).Equiv (right limit target caller)

abbrev Verified (limit target : Int) (caller : Mutable.Heap) :=
  VerifiedCall [integer limit, integer target] limit target caller

private theorem graph_closed (caller : Mutable.Heap) (valid : caller.Valid) :
    ∀ node ∈ (factorials.heap caller).map Mutable.Relocation.graphNode,
      ∀ edge ∈ node.edges, edge.Within (factorials.heap caller).length := by
  intro node member edge edgeMember
  simp only [factorials.heap, List.map_append, List.map_cons, List.map_nil,
    List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with old | rfl
  · have inside := Mutable.Relocation.graphNodes_closed valid node old edge edgeMember
    cases edge with
    | scalar _ => trivial
    | ref address =>
        change address < (factorials.heap caller).length
        change address < caller.length at inside
        simp only [factorials.heap, List.length_append, List.length_cons, List.length_nil]
        omega
  · simp only [Mutable.Relocation.graphNode, factorials.values, List.map_map,
      Function.comp_def, factorials.number, Mutable.Value.edge] at edgeMember
    obtain ⟨value, _, rfl⟩ := List.mem_map.mp edgeMember
    trivial

private theorem globals_within (caller : Mutable.Heap) :
    ∀ entry ∈ factorials.globals caller, entry.2.edge.Within (factorials.heap caller).length := by
  intro entry member
  simp only [factorials.globals, List.mem_cons, List.not_mem_nil, or_false] at member
  subst entry
  change caller.length < (factorials.heap caller).length
  simp [factorials.heap]

/-- Both calls terminate and preserve all caller objects, globals, aliases, and cycles. -/
theorem verified_call (supplied : List Mutable.Value) (limit target : Int)
    (caller : Mutable.Heap) (valid : caller.Valid)
    (filled : completeArguments solveArgs supplied = some [integer limit, integer target]) :
    VerifiedCall supplied limit target caller := by
  obtain ⟨leftExtra, optimized⟩ := solve.initialized_result limit target caller
  have suppliedReturn := ReturnsValue.with_defaults (module := Solvers.P74Ast.module)
    (name := "solve") rfl filled rfl optimized
  obtain ⟨rightExtra, reference⟩ :=
    data.«project-euler-statements».data.py.project_euler_statements.p74.naive.result
      limit target (factorials.heap caller) (factorials.globals caller)
  refine ⟨⟨factorials.executes caller, factorials.exports caller⟩,
    suppliedReturn.returns, reference.returns, ?_⟩
  exact Mutable.Call.equiv_of_finishes suppliedReturn.finishes reference.finishes
    (Mutable.Relocation.equivalent_allocations_of_closed (factorials.heap caller)
      leftExtra rightExtra (integer (ProjectEulerStatements.P74.result limit target))
      (factorials.globals caller) (graph_closed caller valid) trivial (globals_within caller))

theorem verified (limit target : Int) (caller : Mutable.Heap) (valid : caller.Valid) :
    Verified limit target caller :=
  verified_call [integer limit, integer target] limit target caller valid rfl

theorem limit_default_verified (limit : Int) (caller : Mutable.Heap) (valid : caller.Valid) :
    VerifiedCall [integer limit] limit 60 caller :=
  verified_call [integer limit] limit 60 caller valid rfl

theorem all_defaults_verified (caller : Mutable.Heap) (valid : caller.Valid) :
    VerifiedCall [] 1_000_000 60 caller :=
  verified_call [] 1_000_000 60 caller valid rfl

end solvers.«74».source_equivalence
