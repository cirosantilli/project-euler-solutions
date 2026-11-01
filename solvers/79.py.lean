import solvers.«79.py.lean.d».Sequences

/-! The public contract covers every finite list of strings, including invalid attempts and cycles. -/
namespace solvers.«79».source_equivalence
open FormalSW.PyAst
open data.«project-euler-statements».data.py.project_euler_statements
set_option Elab.async false

def optimized (heap : Mutable.Heap) (address : Nat) : Mutable.CallFamily :=
  SetCalls.call graph.program "solve" [.ref address] heap

def reference (heap : Mutable.Heap) (address : Nat) : Mutable.CallFamily :=
  Mutable.CallFamily.deterministic (HeapCalls.call p79.program "naive" [.ref address] heap)

def Verified (heap : Mutable.Heap) (address : Nat) : Prop :=
  (optimized heap address).Halts ∧ (reference heap address).Halts ∧
    (optimized heap address).Supported ∧ (reference heap address).Supported ∧
    (optimized heap address).Equiv (reference heap address)

theorem verified_list (heap : Mutable.Heap) (address : Nat) (attempts : List String)
    (valid : heap.Valid) (found : heap[address]? = some (.list (attempts.map graph.text))) :
    Verified heap address := by
  have input := graph.SequenceInput.list heap address attempts found
  have left (choices : ChoiceSchedule) := source.solve_sequence_finishes choices heap (.ref address) attempts input
  obtain ⟨after, ran, _⟩ := p79.naive_sequence_finishes heap (.ref address) attempts ⟨input.read⟩
  refine ⟨?_, ?_, ?_, ?_, source.sequence_family_equiv heap (.ref address) attempts valid input⟩
  · intro choices
    obtain ⟨_, finished, _⟩ := left choices
    exact ⟨_, finished⟩
  · intro _
    exact ⟨_, ran⟩
  · intro choices
    obtain ⟨_, finished, _⟩ := left choices
    exact Execution.supported_of_finishes finished
  · intro _
    exact Execution.supported_of_finishes ran

end solvers.«79».source_equivalence

namespace solvers.«79»
open FormalSW.PyAst

theorem solve_list_source_equivalence (heap : Mutable.Heap) (address : Nat) (attempts : List String) :
    heap.Valid → heap[address]? = some (.list (attempts.map (fun s => .scalar (.string s)))) →
      source_equivalence.Verified heap address :=
  source_equivalence.verified_list heap address attempts

end solvers.«79»
