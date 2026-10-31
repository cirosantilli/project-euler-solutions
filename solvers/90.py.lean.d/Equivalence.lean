import solvers.«90.py.lean.d».Optimized
import data.«project-euler-statements».data.py.project_euler_statements.«p90.py.lean.d».Execution

namespace solvers.«90».source_equivalence
open FormalSW.PyAst ProjectEulerSolutions
set_option Elab.async false

abbrev ReferencePair := data.«project-euler-statements».data.py.project_euler_statements.p90.can_show.Pair
abbrev Required := data.«project-euler-statements».data.py.project_euler_statements.p90.counting.Required

def optimizedPair (pair : ReferencePair) : can_display.Pair := ⟨pair.x, pair.y, pair.address⟩

/-- Requirements may share tuple objects; all caller references and global aliases remain observable. -/
structure Domain (heap : Mutable.Heap) (globals : Mutable.Globals) (required : Required) : Prop where
  valid : heap.Valid
  globalsWithin : ∀ entry ∈ globals, entry.2.edge.Within heap.length
  requirements : required.Valid heap

def optimizedCall (count : Int) (heap : Mutable.Heap) (globals : Mutable.Globals) (address : Nat) : Mutable.Call :=
  HeapCalls.call program "solve" [.scalar (.int count), .ref address] heap globals

def referenceCall (count : Int) (heap : Mutable.Heap) (globals : Mutable.Globals) (address : Nat) : Mutable.Call :=
  HeapMemoCalls.call P90Ast.module "naive" [.scalar (.int count), .ref address] heap globals

def Verified (count : Int) (heap : Mutable.Heap) (globals : Mutable.Globals) (address : Nat) : Prop :=
  (optimizedCall count heap globals address).Returns ∧ (referenceCall count heap globals address).Returns ∧
    (optimizedCall count heap globals address).Supported ∧ (referenceCall count heap globals address).Supported ∧
    (optimizedCall count heap globals address).Equiv (referenceCall count heap globals address)

/-- Complete source equivalence for every signed count and arbitrary finite nonnegative requirements. -/
theorem verified (count : Int) (heap : Mutable.Heap) (globals : Mutable.Globals) (required : Required)
    (domain : Domain heap globals required) : Verified count heap globals required.address := by
  have found : heap[required.address]? = some (.list ((required.pairs.map optimizedPair).map can_display.Pair.reference)) := by
    simpa only [List.map_map] using domain.requirements.1
  have valid : ∀ pair ∈ required.pairs.map optimizedPair, pair.Valid heap := by
    intro pair member
    obtain ⟨original, originalMember, rfl⟩ := List.mem_map.mp member
    exact domain.requirements.2 original originalMember
  obtain ⟨leftExtra, left⟩ := optimized.signed_result count required.address (required.pairs.map optimizedPair)
    heap globals found valid
  have right := data.«project-euler-statements».data.py.project_euler_statements.p90.naive.result
    count required heap globals domain.requirements
  obtain ⟨value, after, ⟨rfl, rightExtra, rfl⟩, finished⟩ := right.finishes
  refine ⟨left.returns, right.returns, left.returns.supported, right.returns.supported, ?_⟩
  apply Mutable.Call.equiv_of_finishes left.finishes finished
  simpa only [List.map_map] using Mutable.Completion.equivalent_allocations heap leftExtra rightExtra
    (number (DigitDice.answer count (required.pairs.map data.«project-euler-statements».data.py.project_euler_statements.p90.can_show.Pair.model))) globals domain.valid trivial domain.globalsWithin

end solvers.«90».source_equivalence
