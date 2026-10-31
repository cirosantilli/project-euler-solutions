import solvers.«89.py.lean.d».EncoderLoop

namespace solvers.«89»
open FormalSW.PyAst HeapCalls ProjectEulerSolutions RomanNumerals
set_option Elab.async false
set_option maxHeartbeats 1000000

/-- The immutable global table has the standard ordered denominations and token strings. -/
def EncodingTable (heap : Mutable.Heap) (globals : Mutable.Globals) : Prop :=
  ∃ (address : Nat) (entries : List int_to_min_roman.Entry),
    globals.lookup "MINIMAL_TABLE" = some (.ref address) ∧
    heap[address]? = some (.tuple (entries.map int_to_min_roman.Entry.reference)) ∧
    (∀ entry ∈ entries, entry.Valid heap) ∧
    entries.map int_to_min_roman.Entry.model = tokens.map (fun pair => (pair.1, pair.2.toList))

namespace int_to_min_roman

/-- The complete Python encoder emits the reference character model for every positive integer. -/
theorem result (n : Nat) (positive : 0 < n) (heap : Mutable.Heap)
    (globals : Mutable.Globals) (table : EncodingTable heap globals) :
    ∃ extra, ReturnsValue program "int_to_min_roman" [number n] heap
      (textValue (String.ofList (canonical n))) (heap ++ extra) globals := by
  obtain ⟨address, entries, bound, found, valid, models⟩ := table
  let args := ((findFunctionIn program.body "int_to_min_roman").getD ([], [])).1
  let source := ((findFunctionIn program.body "int_to_min_roman").getD ([], [])).2
  let entry := functionEnv args source [number n] globals
  have getN : entry "x" = some (number n) := by cases globals <;> rfl
  have getTable : entry "MINIMAL_TABLE" = some (.ref address) := by
    have read : entry "MINIMAL_TABLE" = globals.lookup "MINIMAL_TABLE" := by
      cases globals with
      | nil => rfl
      | cons item rest =>
          have notArgument : "MINIMAL_TABLE" ∉ args.map Arg.name := by decide
          have notLocal : "MINIMAL_TABLE" ∉ blockLocals source := by
            simp [source, program, Solvers.P89Ast.module, findFunctionIn, blockLocals, assignedLocals, targetLocals]
          simp only [entry, functionEnv, notArgument, notLocal, if_false]
    exact read.trans bound
  let ready := setLocal entry "res" (.ref heap.length)
  have allocated : BlockContinues program entry [code[1]'(by decide)] heap ready (working heap [] []) globals :=
    BlockContinues.annAssign (Evaluates.list .nil)
  obtain ⟨output, garbage, scanned⟩ := table_loop heap address entries entries found valid 0 rfl n [] [] globals ready
    (by simpa [ready, setLocal] using getN) (by simp [ready, setLocal])
  have modeled : encode (entries.map Entry.model) n = canonical n := by rw [models]; rfl
  refine ⟨.list (output.map textValue) :: garbage, ?_⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.branch (flag := false) (value := .scalar (.bool false)) (middle := heap)
  · apply Evaluates.compare (Evaluates.lookup getN) Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    omega
  · rfl
  apply allocated.returns
  apply BlockReturns.for_unpack (items := entries.map Entry.reference) (middle := working heap [] [])
    (by intros; simp) rfl (Evaluates.lookup (by simpa [ready, setLocal] using getTable))
  · have table := HeapCollection.prefix_found heap [.list []] address _ found
    simp [Mutable.sequence, working, HeapCollection.working, table]
  · simpa only [List.flatMap_nil, List.nil_append, modeled] using scanned

end int_to_min_roman
end solvers.«89»
