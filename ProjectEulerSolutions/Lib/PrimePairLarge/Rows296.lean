import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row9619_small : List Nat := []
theorem row9619_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9619 row9619_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9619_block_005 : List Nat := [10039, 10333]
def plan9619_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .second, (.secondFactor 19), (.firstFactor 17), .first, .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 29), .first, .second, (.secondFactor 13), (.secondFactor 7), .first, (.secondFactor 19), .first, .possible, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 19), (.firstFactor 17), .first, .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 17), .first, (.firstFactor 19), .first, .possible, .first, .first, (.secondFactor 11), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row9619_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9619 row9619_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9619_block_005
  · decide
  · rfl

def row9619_block_006 : List Nat := [10993, 11239, 11329, 11959, 11971, 12253, 12379]
def plan9619_block_006 : List Budget.Choice := [.first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 31), (.secondFactor 13), .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 13), (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 19), (.secondFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .possible, .first, .first, .second, .first, (.firstFactor 31), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 31), .first, (.firstFactor 13), .second, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .possible, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 37), (.secondFactor 7), .first, .possible, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 19), (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 17), .first, .second, .first, (.firstFactor 7), (.secondFactor 23), .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 29), .first, (.secondFactor 19), (.firstFactor 11), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .possible, .first, .possible, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 31), (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 29), .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .possible, .first, .first, .second, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .possible, (.firstFactor 11), .first, (.secondFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row9619_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9619 row9619_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9619_block_006
  · decide
  · rfl

def row9619_block_007 : List Nat := [13003, 13759, 13831]
def plan9619_block_007 : List Budget.Choice := [(.firstFactor 37), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 13), (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 17), (.firstFactor 19), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 7), .first, .first, .first, .possible, .first, (.secondFactor 13), .second, .first, .first, .first, (.secondFactor 11), (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .second, (.secondFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 19), (.secondFactor 13), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .possible, .first, .first, (.firstFactor 29), .first, .first, .first, .possible, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.secondFactor 31), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 19), .first, (.secondFactor 29), .second, .first, .first, .first, .second, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 31), .first, .first, (.secondFactor 11), (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 37), .first, .first, (.firstFactor 11), .first, (.secondFactor 11), (.secondFactor 31)]
set_option maxRecDepth 2048 in
theorem row9619_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9619 row9619_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9619_block_007
  · decide
  · rfl

def row9619_block_008 : List Nat := [14827, 16369]
def plan9619_block_008 : List Budget.Choice := [.first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 13), (.secondFactor 17), .first, .second, .first, (.firstFactor 31), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.secondFactor 11), .first, .first, .second, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 7), .possible, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .second, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .second, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 37), .first, (.secondFactor 23), .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, (.secondFactor 23), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), (.secondFactor 11), (.firstFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 37), .second, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 19), .possible, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9619_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9619 row9619_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9619_block_008
  · decide
  · rfl

def row9619_block_009 : List Nat := []
theorem row9619_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9619 row9619_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9619_block_010 : List Nat := []
theorem row9619_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9619 row9619_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9619_block_011 : List Nat := []
theorem row9619_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9619 row9619_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9619_block_012 : List Nat := []
theorem row9619_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9619 row9619_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9619_block_013 : List Nat := []
theorem row9619_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9619 row9619_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9619_large_left_left : List Nat := row9619_small ++ row9619_block_005
theorem row9619_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9619 row9619_large_left_left :=
  Budget.RowUpperBound.branch row9619_small_covers row9619_block_005_covers

def row9619_large_left_right_right : List Nat := row9619_block_007 ++ row9619_block_008
theorem row9619_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9619 row9619_large_left_right_right :=
  Budget.RowUpperBound.branch row9619_block_007_covers row9619_block_008_covers

def row9619_large_left_right : List Nat := row9619_block_006 ++ row9619_large_left_right_right
theorem row9619_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9619 row9619_large_left_right :=
  Budget.RowUpperBound.branch row9619_block_006_covers row9619_large_left_right_right_covers

def row9619_large_left : List Nat := row9619_large_left_left ++ row9619_large_left_right
theorem row9619_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9619 row9619_large_left :=
  Budget.RowUpperBound.branch row9619_large_left_left_covers row9619_large_left_right_covers

def row9619_large_right_left : List Nat := row9619_block_009 ++ row9619_block_010
theorem row9619_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9619 row9619_large_right_left :=
  Budget.RowUpperBound.branch row9619_block_009_covers row9619_block_010_covers

def row9619_large_right_right_right : List Nat := row9619_block_012 ++ row9619_block_013
theorem row9619_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9619 row9619_large_right_right_right :=
  Budget.RowUpperBound.branch row9619_block_012_covers row9619_block_013_covers

def row9619_large_right_right : List Nat := row9619_block_011 ++ row9619_large_right_right_right
theorem row9619_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9619 row9619_large_right_right :=
  Budget.RowUpperBound.branch row9619_block_011_covers row9619_large_right_right_right_covers

def row9619_large_right : List Nat := row9619_large_right_left ++ row9619_large_right_right
theorem row9619_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9619 row9619_large_right :=
  Budget.RowUpperBound.branch row9619_large_right_left_covers row9619_large_right_right_covers

def row9619_large : List Nat := row9619_large_left ++ row9619_large_right
theorem row9619_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9619 row9619_large :=
  Budget.RowUpperBound.branch row9619_large_left_covers row9619_large_right_covers

def row9619 : List Nat := row9619_large
theorem row9619_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9619 row9619 := row9619_large_covers

def row9623_small : List Nat := []
theorem row9623_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9623 row9623_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9623_block_005 : List Nat := []
def plan9623_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 29), (.secondFactor 13), (.firstFactor 13), .first, .first, .first, .second, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), .second, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, (.secondFactor 13), (.secondFactor 11), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 29), .first, (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 31), .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 17), (.firstFactor 7), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 11), .first, .second, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9623_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9623 row9623_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9623_block_005
  · decide
  · rfl

def row9623_block_006 : List Nat := [10733, 10949, 12347]
def plan9623_block_006 : List Budget.Choice := [(.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .second, .first, .first, .first, .first, .possible, (.firstFactor 13), .first, .first, (.secondFactor 19), .first, (.firstFactor 29), .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 31), (.firstFactor 7), (.firstFactor 19), .second, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .second, .first, (.firstFactor 31), .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, (.secondFactor 29), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 23), (.firstFactor 11), (.secondFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 23), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .possible, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9623_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9623 row9623_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9623_block_006
  · decide
  · rfl

def row9623_block_007 : List Nat := [13151, 14009]
def plan9623_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 11), (.firstFactor 29), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 11), .second, (.secondFactor 7), (.firstFactor 7), .second, (.firstFactor 13), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 17), (.firstFactor 7), (.secondFactor 31), .first, .first, .first, (.firstFactor 37), (.firstFactor 13), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 17), (.secondFactor 29), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 19), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.firstFactor 19), .first, (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 29), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 23), .first, .possible, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .second, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.secondFactor 31), .first, .first]
set_option maxRecDepth 2048 in
theorem row9623_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9623 row9623_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9623_block_007
  · decide
  · rfl

def row9623_block_008 : List Nat := [15359, 15461, 15527, 15581, 15737, 16217]
def plan9623_block_008 : List Budget.Choice := [.first, (.firstFactor 19), .first, .first, .first, (.firstFactor 37), (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 19), (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, .second, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, .second, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 11), (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 23), .first, .first, .possible, .first, .first, (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 17), .first, .first, (.secondFactor 13), .first, .possible, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, .possible, .first, (.secondFactor 7), .first, (.secondFactor 19), .possible, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .possible, .first, (.secondFactor 31), (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 23), (.secondFactor 17), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 19), .first, .first, (.secondFactor 17), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 11), .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .possible, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 13), .second, .first, .first, (.secondFactor 7), .second, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9623_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9623 row9623_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9623_block_008
  · decide
  · rfl

def row9623_block_009 : List Nat := []
theorem row9623_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9623 row9623_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9623_block_010 : List Nat := []
theorem row9623_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9623 row9623_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9623_block_011 : List Nat := []
theorem row9623_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9623 row9623_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9623_block_012 : List Nat := []
theorem row9623_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9623 row9623_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9623_block_013 : List Nat := []
theorem row9623_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9623 row9623_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9623_large_left_left : List Nat := row9623_small ++ row9623_block_005
theorem row9623_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9623 row9623_large_left_left :=
  Budget.RowUpperBound.branch row9623_small_covers row9623_block_005_covers

def row9623_large_left_right_right : List Nat := row9623_block_007 ++ row9623_block_008
theorem row9623_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9623 row9623_large_left_right_right :=
  Budget.RowUpperBound.branch row9623_block_007_covers row9623_block_008_covers

def row9623_large_left_right : List Nat := row9623_block_006 ++ row9623_large_left_right_right
theorem row9623_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9623 row9623_large_left_right :=
  Budget.RowUpperBound.branch row9623_block_006_covers row9623_large_left_right_right_covers

def row9623_large_left : List Nat := row9623_large_left_left ++ row9623_large_left_right
theorem row9623_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9623 row9623_large_left :=
  Budget.RowUpperBound.branch row9623_large_left_left_covers row9623_large_left_right_covers

def row9623_large_right_left : List Nat := row9623_block_009 ++ row9623_block_010
theorem row9623_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9623 row9623_large_right_left :=
  Budget.RowUpperBound.branch row9623_block_009_covers row9623_block_010_covers

def row9623_large_right_right_right : List Nat := row9623_block_012 ++ row9623_block_013
theorem row9623_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9623 row9623_large_right_right_right :=
  Budget.RowUpperBound.branch row9623_block_012_covers row9623_block_013_covers

def row9623_large_right_right : List Nat := row9623_block_011 ++ row9623_large_right_right_right
theorem row9623_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9623 row9623_large_right_right :=
  Budget.RowUpperBound.branch row9623_block_011_covers row9623_large_right_right_right_covers

def row9623_large_right : List Nat := row9623_large_right_left ++ row9623_large_right_right
theorem row9623_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9623 row9623_large_right :=
  Budget.RowUpperBound.branch row9623_large_right_left_covers row9623_large_right_right_covers

def row9623_large : List Nat := row9623_large_left ++ row9623_large_right
theorem row9623_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9623 row9623_large :=
  Budget.RowUpperBound.branch row9623_large_left_covers row9623_large_right_covers

def row9623 : List Nat := row9623_large
theorem row9623_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9623 row9623 := row9623_large_covers

def row9629_small : List Nat := []
theorem row9629_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9629 row9629_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9629_block_005 : List Nat := []
def plan9629_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, .second, .first, (.firstFactor 23), .first, (.secondFactor 37), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .second, (.secondFactor 13), (.firstFactor 31), .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 19), (.firstFactor 11), .first, (.secondFactor 37), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 13), .first, .second, (.firstFactor 7), (.secondFactor 17), (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 19), (.secondFactor 11), .first, .first, .first, (.firstFactor 11), (.secondFactor 29), .first, (.firstFactor 31), .first, (.secondFactor 7), .first, .second, (.secondFactor 19), .first, .first]
set_option maxRecDepth 2048 in
theorem row9629_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9629 row9629_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9629_block_005
  · decide
  · rfl

def row9629_block_006 : List Nat := [10979, 11351]
def plan9629_block_006 : List Budget.Choice := [(.secondFactor 11), .first, (.secondFactor 31), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, (.firstFactor 37), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 23), (.secondFactor 11), (.secondFactor 23), .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .possible, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 29), (.firstFactor 11), .first, .first, .first, .first, .second, .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 17), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 19), (.secondFactor 13), .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .second, .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 11), .first, .second, (.firstFactor 7), (.firstFactor 31), (.firstFactor 11), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 29), .first, .first, .second, (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 31), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 23), (.secondFactor 11), .first, (.firstFactor 17), (.firstFactor 13), (.firstFactor 29), (.secondFactor 17), .first, (.firstFactor 37), .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 23), (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row9629_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9629 row9629_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9629_block_006
  · decide
  · rfl

def row9629_block_007 : List Nat := [13109, 13463, 13613, 14177, 14423]
def plan9629_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .second, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 23), (.firstFactor 19), (.secondFactor 23), (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 29), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .possible, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 7), .second, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 17), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 19), .first, .second, (.firstFactor 11), .possible, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .possible, (.secondFactor 23), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 17), .first, .second, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 23), (.firstFactor 19), .first, (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 17), (.firstFactor 23), .first, .possible, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .possible, .first, .first, (.firstFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row9629_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9629 row9629_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9629_block_007
  · decide
  · rfl

def row9629_block_008 : List Nat := []
def plan9629_block_008 : List Budget.Choice := [.first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .second, .second, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 7), (.secondFactor 11), (.firstFactor 29), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 11), .first, (.secondFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .second, (.firstFactor 17), .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 19), (.secondFactor 23), .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 29), (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 29), .first, .first, (.secondFactor 7), (.secondFactor 31), .first, .first, .second, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 11), (.secondFactor 23), .first, (.secondFactor 7), .second, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 29), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9629_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9629 row9629_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9629_block_008
  · decide
  · rfl

def row9629_block_009 : List Nat := []
theorem row9629_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9629 row9629_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9629_block_010 : List Nat := []
theorem row9629_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9629 row9629_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9629_block_011 : List Nat := []
theorem row9629_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9629 row9629_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9629_block_012 : List Nat := []
theorem row9629_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9629 row9629_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9629_block_013 : List Nat := []
theorem row9629_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9629 row9629_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9629_large_left_left : List Nat := row9629_small ++ row9629_block_005
theorem row9629_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9629 row9629_large_left_left :=
  Budget.RowUpperBound.branch row9629_small_covers row9629_block_005_covers

def row9629_large_left_right_right : List Nat := row9629_block_007 ++ row9629_block_008
theorem row9629_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9629 row9629_large_left_right_right :=
  Budget.RowUpperBound.branch row9629_block_007_covers row9629_block_008_covers

def row9629_large_left_right : List Nat := row9629_block_006 ++ row9629_large_left_right_right
theorem row9629_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9629 row9629_large_left_right :=
  Budget.RowUpperBound.branch row9629_block_006_covers row9629_large_left_right_right_covers

def row9629_large_left : List Nat := row9629_large_left_left ++ row9629_large_left_right
theorem row9629_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9629 row9629_large_left :=
  Budget.RowUpperBound.branch row9629_large_left_left_covers row9629_large_left_right_covers

def row9629_large_right_left : List Nat := row9629_block_009 ++ row9629_block_010
theorem row9629_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9629 row9629_large_right_left :=
  Budget.RowUpperBound.branch row9629_block_009_covers row9629_block_010_covers

def row9629_large_right_right_right : List Nat := row9629_block_012 ++ row9629_block_013
theorem row9629_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9629 row9629_large_right_right_right :=
  Budget.RowUpperBound.branch row9629_block_012_covers row9629_block_013_covers

def row9629_large_right_right : List Nat := row9629_block_011 ++ row9629_large_right_right_right
theorem row9629_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9629 row9629_large_right_right :=
  Budget.RowUpperBound.branch row9629_block_011_covers row9629_large_right_right_right_covers

def row9629_large_right : List Nat := row9629_large_right_left ++ row9629_large_right_right
theorem row9629_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9629 row9629_large_right :=
  Budget.RowUpperBound.branch row9629_large_right_left_covers row9629_large_right_right_covers

def row9629_large : List Nat := row9629_large_left ++ row9629_large_right
theorem row9629_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9629 row9629_large :=
  Budget.RowUpperBound.branch row9629_large_left_covers row9629_large_right_covers

def row9629 : List Nat := row9629_large
theorem row9629_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9629 row9629 := row9629_large_covers

def row9631_small : List Nat := []
theorem row9631_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9631 row9631_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9631_block_005 : List Nat := [9679]
def plan9631_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 7), .first, .second, .first, .possible, .first, (.firstFactor 31), .first, (.firstFactor 19), (.secondFactor 29), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 23), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 11), (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 29), .first, .first, .second, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 23), .first, (.secondFactor 17), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row9631_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9631 row9631_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9631_block_005
  · decide
  · rfl

def row9631_block_006 : List Nat := [10903, 11059, 11833, 12289]
def plan9631_block_006 : List Budget.Choice := [.first, (.firstFactor 19), .first, (.secondFactor 29), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 17), .first, (.secondFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .possible, (.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .possible, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 17), .first, (.firstFactor 17), (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .second, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 31), .first, .first, .first, .second, (.secondFactor 11), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .possible, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 29), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 23), .first, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, .possible, (.firstFactor 29), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .second, (.secondFactor 11), .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row9631_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9631 row9631_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9631_block_006
  · decide
  · rfl

def row9631_block_007 : List Nat := [13147, 13999]
def plan9631_block_007 : List Budget.Choice := [.first, (.firstFactor 31), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 29), .first, .first, .second, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .second, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 23), .first, (.secondFactor 19), .second, .first, .first, .first, (.firstFactor 11), .second, (.firstFactor 13), .first, .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 29), (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 23), .first, (.secondFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 11), .first, .second, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, (.secondFactor 19), .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .possible, .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 29), .first, (.firstFactor 17), .second, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .second, .first, .first, (.firstFactor 11), .second, (.firstFactor 31), .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, (.secondFactor 31), .first, (.secondFactor 29), (.secondFactor 11), .first, (.firstFactor 11), (.secondFactor 13)]
set_option maxRecDepth 2048 in
theorem row9631_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9631 row9631_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9631_block_007
  · decide
  · rfl

def row9631_block_008 : List Nat := [15583, 16069]
def plan9631_block_008 : List Budget.Choice := [(.firstFactor 31), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 11), .first, .first, (.firstFactor 19), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 11), .first, (.secondFactor 23), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 19), (.firstFactor 19), .first, .first, .second, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 11), (.firstFactor 29), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, .possible, (.firstFactor 7), (.firstFactor 29), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 29), (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 19), .first, .first, .first, (.secondFactor 11), (.firstFactor 11), (.firstFactor 13), (.secondFactor 7), .second, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .possible, .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 11), .first, .second, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9631_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9631 row9631_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9631_block_008
  · decide
  · rfl

def row9631_block_009 : List Nat := []
theorem row9631_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9631 row9631_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9631_block_010 : List Nat := []
theorem row9631_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9631 row9631_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9631_block_011 : List Nat := []
theorem row9631_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9631 row9631_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9631_block_012 : List Nat := []
theorem row9631_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9631 row9631_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9631_block_013 : List Nat := []
theorem row9631_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9631 row9631_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9631_large_left_left : List Nat := row9631_small ++ row9631_block_005
theorem row9631_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9631 row9631_large_left_left :=
  Budget.RowUpperBound.branch row9631_small_covers row9631_block_005_covers

def row9631_large_left_right_right : List Nat := row9631_block_007 ++ row9631_block_008
theorem row9631_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9631 row9631_large_left_right_right :=
  Budget.RowUpperBound.branch row9631_block_007_covers row9631_block_008_covers

def row9631_large_left_right : List Nat := row9631_block_006 ++ row9631_large_left_right_right
theorem row9631_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9631 row9631_large_left_right :=
  Budget.RowUpperBound.branch row9631_block_006_covers row9631_large_left_right_right_covers

def row9631_large_left : List Nat := row9631_large_left_left ++ row9631_large_left_right
theorem row9631_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9631 row9631_large_left :=
  Budget.RowUpperBound.branch row9631_large_left_left_covers row9631_large_left_right_covers

def row9631_large_right_left : List Nat := row9631_block_009 ++ row9631_block_010
theorem row9631_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9631 row9631_large_right_left :=
  Budget.RowUpperBound.branch row9631_block_009_covers row9631_block_010_covers

def row9631_large_right_right_right : List Nat := row9631_block_012 ++ row9631_block_013
theorem row9631_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9631 row9631_large_right_right_right :=
  Budget.RowUpperBound.branch row9631_block_012_covers row9631_block_013_covers

def row9631_large_right_right : List Nat := row9631_block_011 ++ row9631_large_right_right_right
theorem row9631_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9631 row9631_large_right_right :=
  Budget.RowUpperBound.branch row9631_block_011_covers row9631_large_right_right_right_covers

def row9631_large_right : List Nat := row9631_large_right_left ++ row9631_large_right_right
theorem row9631_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9631 row9631_large_right :=
  Budget.RowUpperBound.branch row9631_large_right_left_covers row9631_large_right_right_covers

def row9631_large : List Nat := row9631_large_left ++ row9631_large_right
theorem row9631_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9631 row9631_large :=
  Budget.RowUpperBound.branch row9631_large_left_covers row9631_large_right_covers

def row9631 : List Nat := row9631_large
theorem row9631_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9631 row9631 := row9631_large_covers

end ProjectEulerSolutions.PrimePairLarge
