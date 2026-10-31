import solvers.«89.py.lean.d».SolverExecution
import data.«project-euler-statements».data.py.project_euler_statements.«p89.py.lean.d».Execution

namespace solvers.«89».source_equivalence
open FormalSW.PyAst ProjectEulerSolutions
set_option Elab.async false

/-- The generalized problem domain and the standard tables at call entry. -/
structure Domain (heap : Mutable.Heap) (globals : Mutable.Globals) (address : Nat) (lines : List String) : Prop where
  valid : heap.Valid
  bindings : ∀ entry ∈ globals, entry.1 = "ROMAN_VALUE" ∨ entry.1 = "MINIMAL_TABLE"
  globalsWithin : ∀ entry ∈ globals, entry.2.edge.Within heap.length
  input : heap[address]? = some (.list (lines.map textValue))
  texts : ∀ text ∈ lines, ValidText text
  symbols : SymbolTable heap globals
  tokens : EncodingTable heap globals

def optimized (heap : Mutable.Heap) (globals : Mutable.Globals) (address : Nat) : Mutable.Call :=
  HeapCalls.call program "solve" [.ref address] heap globals

def reference (heap : Mutable.Heap) (globals : Mutable.Globals) (address : Nat) : Mutable.Call :=
  HeapCalls.call P89Ast.module "naive" [.ref address] heap globals

def Verified (heap : Mutable.Heap) (globals : Mutable.Globals) (address : Nat) : Prop :=
  (optimized heap globals address).Returns ∧ (reference heap globals address).Returns ∧
    (optimized heap globals address).Supported ∧ (reference heap globals address).Supported ∧
    (optimized heap globals address).Equiv (reference heap globals address)

/-- Both actual implementations preserve caller objects and global aliases while returning the same sum. -/
theorem verified (heap : Mutable.Heap) (globals : Mutable.Globals) (address : Nat) (lines : List String)
    (domain : Domain heap globals address lines) : Verified heap globals address := by
  obtain ⟨leftExtra, left⟩ := solve.result lines domain.texts heap globals domain.symbols domain.tokens address domain.input
  obtain ⟨rightExtra, right⟩ :=
    data.«project-euler-statements».data.py.project_euler_statements.p89.naive.result
      (globals := globals) lines heap address domain.input
  refine ⟨left.returns, right.returns, left.returns.supported, right.returns.supported, ?_⟩
  apply Mutable.Call.equiv_of_finishes left.finishes right.finishes
  exact Mutable.Completion.equivalent_allocations heap leftExtra rightExtra
    (.scalar (.int (RomanNumerals.savings lines))) globals domain.valid trivial domain.globalsWithin

end solvers.«89».source_equivalence
