import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row12391_small : List Nat := []
theorem row12391_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12391 row12391_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12391_block_005 : List Nat := []
theorem row12391_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12391 row12391_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12391_block_006 : List Nat := []
def plan12391_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row12391_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12391 row12391_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12391_block_006
  · decide
  · rfl

def row12391_block_007 : List Nat := [12517, 12781, 13627]
def plan12391_block_007 : List Budget.Choice := [(.secondFactor 13), (.firstFactor 13), .first, .second, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .possible, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 7), .possible, .first, .first, .first, .first, (.firstFactor 13), .second, (.firstFactor 29), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, .second, .first, (.secondFactor 7), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 23), .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 29), .first, .second, .first, (.secondFactor 7), .second, .first, .first, .possible, (.firstFactor 19), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12391_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12391 row12391_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12391_block_007
  · decide
  · rfl

def row12391_block_008 : List Nat := []
theorem row12391_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12391 row12391_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12391_block_009 : List Nat := []
theorem row12391_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12391 row12391_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12391_block_010 : List Nat := []
theorem row12391_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12391 row12391_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12391_block_011 : List Nat := []
theorem row12391_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12391 row12391_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12391_block_012 : List Nat := []
theorem row12391_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12391 row12391_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12391_block_013 : List Nat := []
theorem row12391_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12391 row12391_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12391_large_left_left : List Nat := row12391_small ++ row12391_block_005
theorem row12391_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12391 row12391_large_left_left :=
  Budget.RowUpperBound.branch row12391_small_covers row12391_block_005_covers

def row12391_large_left_right_right : List Nat := row12391_block_007 ++ row12391_block_008
theorem row12391_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12391 row12391_large_left_right_right :=
  Budget.RowUpperBound.branch row12391_block_007_covers row12391_block_008_covers

def row12391_large_left_right : List Nat := row12391_block_006 ++ row12391_large_left_right_right
theorem row12391_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12391 row12391_large_left_right :=
  Budget.RowUpperBound.branch row12391_block_006_covers row12391_large_left_right_right_covers

def row12391_large_left : List Nat := row12391_large_left_left ++ row12391_large_left_right
theorem row12391_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12391 row12391_large_left :=
  Budget.RowUpperBound.branch row12391_large_left_left_covers row12391_large_left_right_covers

def row12391_large_right_left : List Nat := row12391_block_009 ++ row12391_block_010
theorem row12391_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12391 row12391_large_right_left :=
  Budget.RowUpperBound.branch row12391_block_009_covers row12391_block_010_covers

def row12391_large_right_right_right : List Nat := row12391_block_012 ++ row12391_block_013
theorem row12391_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12391 row12391_large_right_right_right :=
  Budget.RowUpperBound.branch row12391_block_012_covers row12391_block_013_covers

def row12391_large_right_right : List Nat := row12391_block_011 ++ row12391_large_right_right_right
theorem row12391_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12391 row12391_large_right_right :=
  Budget.RowUpperBound.branch row12391_block_011_covers row12391_large_right_right_right_covers

def row12391_large_right : List Nat := row12391_large_right_left ++ row12391_large_right_right
theorem row12391_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12391 row12391_large_right :=
  Budget.RowUpperBound.branch row12391_large_right_left_covers row12391_large_right_right_covers

def row12391_large : List Nat := row12391_large_left ++ row12391_large_right
theorem row12391_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12391 row12391_large :=
  Budget.RowUpperBound.branch row12391_large_left_covers row12391_large_right_covers

def row12391 : List Nat := row12391_large
theorem row12391_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12391 row12391 := row12391_large_covers

def row12401_small : List Nat := []
theorem row12401_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12401 row12401_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12401_block_005 : List Nat := []
theorem row12401_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12401 row12401_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12401_block_006 : List Nat := []
def plan12401_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row12401_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12401 row12401_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12401_block_006
  · decide
  · rfl

def row12401_block_007 : List Nat := [12611, 13499]
def plan12401_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), (.secondFactor 13), (.firstFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 13), .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 19), (.secondFactor 31), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 19), (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 19), .first, (.firstFactor 31), .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .second, .possible, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 29), .first, .first, .second, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12401_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12401 row12401_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12401_block_007
  · decide
  · rfl

def row12401_block_008 : List Nat := []
theorem row12401_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12401 row12401_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12401_block_009 : List Nat := []
theorem row12401_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12401 row12401_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12401_block_010 : List Nat := []
theorem row12401_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12401 row12401_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12401_block_011 : List Nat := []
theorem row12401_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12401 row12401_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12401_block_012 : List Nat := []
theorem row12401_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12401 row12401_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12401_block_013 : List Nat := []
theorem row12401_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12401 row12401_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12401_large_left_left : List Nat := row12401_small ++ row12401_block_005
theorem row12401_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12401 row12401_large_left_left :=
  Budget.RowUpperBound.branch row12401_small_covers row12401_block_005_covers

def row12401_large_left_right_right : List Nat := row12401_block_007 ++ row12401_block_008
theorem row12401_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12401 row12401_large_left_right_right :=
  Budget.RowUpperBound.branch row12401_block_007_covers row12401_block_008_covers

def row12401_large_left_right : List Nat := row12401_block_006 ++ row12401_large_left_right_right
theorem row12401_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12401 row12401_large_left_right :=
  Budget.RowUpperBound.branch row12401_block_006_covers row12401_large_left_right_right_covers

def row12401_large_left : List Nat := row12401_large_left_left ++ row12401_large_left_right
theorem row12401_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12401 row12401_large_left :=
  Budget.RowUpperBound.branch row12401_large_left_left_covers row12401_large_left_right_covers

def row12401_large_right_left : List Nat := row12401_block_009 ++ row12401_block_010
theorem row12401_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12401 row12401_large_right_left :=
  Budget.RowUpperBound.branch row12401_block_009_covers row12401_block_010_covers

def row12401_large_right_right_right : List Nat := row12401_block_012 ++ row12401_block_013
theorem row12401_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12401 row12401_large_right_right_right :=
  Budget.RowUpperBound.branch row12401_block_012_covers row12401_block_013_covers

def row12401_large_right_right : List Nat := row12401_block_011 ++ row12401_large_right_right_right
theorem row12401_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12401 row12401_large_right_right :=
  Budget.RowUpperBound.branch row12401_block_011_covers row12401_large_right_right_right_covers

def row12401_large_right : List Nat := row12401_large_right_left ++ row12401_large_right_right
theorem row12401_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12401 row12401_large_right :=
  Budget.RowUpperBound.branch row12401_large_right_left_covers row12401_large_right_right_covers

def row12401_large : List Nat := row12401_large_left ++ row12401_large_right
theorem row12401_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12401 row12401_large :=
  Budget.RowUpperBound.branch row12401_large_left_covers row12401_large_right_covers

def row12401 : List Nat := row12401_large
theorem row12401_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12401 row12401 := row12401_large_covers

def row12409_small : List Nat := []
theorem row12409_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12409 row12409_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12409_block_005 : List Nat := []
theorem row12409_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12409 row12409_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12409_block_006 : List Nat := []
def plan12409_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first]
set_option maxRecDepth 2048 in
theorem row12409_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12409 row12409_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12409_block_006
  · decide
  · rfl

def row12409_block_007 : List Nat := [13309, 13417]
def plan12409_block_007 : List Budget.Choice := [(.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 19), .first, .second, (.firstFactor 23), (.firstFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 29), (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 23), (.firstFactor 11), .first, (.firstFactor 7), .second, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 31), (.firstFactor 13), .possible, .first, (.secondFactor 7), .first, .first, .first, .first, .second, .first, (.firstFactor 11), (.firstFactor 23), .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12409_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12409 row12409_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12409_block_007
  · decide
  · rfl

def row12409_block_008 : List Nat := []
theorem row12409_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12409 row12409_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12409_block_009 : List Nat := []
theorem row12409_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12409 row12409_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12409_block_010 : List Nat := []
theorem row12409_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12409 row12409_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12409_block_011 : List Nat := []
theorem row12409_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12409 row12409_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12409_block_012 : List Nat := []
theorem row12409_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12409 row12409_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12409_block_013 : List Nat := []
theorem row12409_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12409 row12409_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12409_large_left_left : List Nat := row12409_small ++ row12409_block_005
theorem row12409_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12409 row12409_large_left_left :=
  Budget.RowUpperBound.branch row12409_small_covers row12409_block_005_covers

def row12409_large_left_right_right : List Nat := row12409_block_007 ++ row12409_block_008
theorem row12409_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12409 row12409_large_left_right_right :=
  Budget.RowUpperBound.branch row12409_block_007_covers row12409_block_008_covers

def row12409_large_left_right : List Nat := row12409_block_006 ++ row12409_large_left_right_right
theorem row12409_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12409 row12409_large_left_right :=
  Budget.RowUpperBound.branch row12409_block_006_covers row12409_large_left_right_right_covers

def row12409_large_left : List Nat := row12409_large_left_left ++ row12409_large_left_right
theorem row12409_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12409 row12409_large_left :=
  Budget.RowUpperBound.branch row12409_large_left_left_covers row12409_large_left_right_covers

def row12409_large_right_left : List Nat := row12409_block_009 ++ row12409_block_010
theorem row12409_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12409 row12409_large_right_left :=
  Budget.RowUpperBound.branch row12409_block_009_covers row12409_block_010_covers

def row12409_large_right_right_right : List Nat := row12409_block_012 ++ row12409_block_013
theorem row12409_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12409 row12409_large_right_right_right :=
  Budget.RowUpperBound.branch row12409_block_012_covers row12409_block_013_covers

def row12409_large_right_right : List Nat := row12409_block_011 ++ row12409_large_right_right_right
theorem row12409_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12409 row12409_large_right_right :=
  Budget.RowUpperBound.branch row12409_block_011_covers row12409_large_right_right_right_covers

def row12409_large_right : List Nat := row12409_large_right_left ++ row12409_large_right_right
theorem row12409_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12409 row12409_large_right :=
  Budget.RowUpperBound.branch row12409_large_right_left_covers row12409_large_right_right_covers

def row12409_large : List Nat := row12409_large_left ++ row12409_large_right
theorem row12409_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12409 row12409_large :=
  Budget.RowUpperBound.branch row12409_large_left_covers row12409_large_right_covers

def row12409 : List Nat := row12409_large
theorem row12409_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12409 row12409 := row12409_large_covers

def row12413_small : List Nat := []
theorem row12413_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12413 row12413_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12413_block_005 : List Nat := []
theorem row12413_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12413 row12413_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12413_block_006 : List Nat := []
def plan12413_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12413_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12413 row12413_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12413_block_006
  · decide
  · rfl

def row12413_block_007 : List Nat := [12473, 12497, 12641, 12791, 13043, 13613]
def plan12413_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 37), .first, .first, .possible, (.firstFactor 11), .first, (.firstFactor 29), .possible, (.firstFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .possible, (.firstFactor 17), (.firstFactor 7), (.secondFactor 37), (.secondFactor 7), .second, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .possible, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 31), (.secondFactor 19), .first, (.secondFactor 17), (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 17), (.firstFactor 19), .first, .first, .first, .second, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 37), .possible, (.secondFactor 7), .first, .first, .first, (.secondFactor 37), .first, .first, (.secondFactor 19), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .second, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .possible, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12413_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12413 row12413_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12413_block_007
  · decide
  · rfl

def row12413_block_008 : List Nat := []
theorem row12413_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12413 row12413_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12413_block_009 : List Nat := []
theorem row12413_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12413 row12413_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12413_block_010 : List Nat := []
theorem row12413_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12413 row12413_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12413_block_011 : List Nat := []
theorem row12413_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12413 row12413_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12413_block_012 : List Nat := []
theorem row12413_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12413 row12413_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12413_block_013 : List Nat := []
theorem row12413_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12413 row12413_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12413_large_left_left : List Nat := row12413_small ++ row12413_block_005
theorem row12413_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12413 row12413_large_left_left :=
  Budget.RowUpperBound.branch row12413_small_covers row12413_block_005_covers

def row12413_large_left_right_right : List Nat := row12413_block_007 ++ row12413_block_008
theorem row12413_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12413 row12413_large_left_right_right :=
  Budget.RowUpperBound.branch row12413_block_007_covers row12413_block_008_covers

def row12413_large_left_right : List Nat := row12413_block_006 ++ row12413_large_left_right_right
theorem row12413_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12413 row12413_large_left_right :=
  Budget.RowUpperBound.branch row12413_block_006_covers row12413_large_left_right_right_covers

def row12413_large_left : List Nat := row12413_large_left_left ++ row12413_large_left_right
theorem row12413_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12413 row12413_large_left :=
  Budget.RowUpperBound.branch row12413_large_left_left_covers row12413_large_left_right_covers

def row12413_large_right_left : List Nat := row12413_block_009 ++ row12413_block_010
theorem row12413_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12413 row12413_large_right_left :=
  Budget.RowUpperBound.branch row12413_block_009_covers row12413_block_010_covers

def row12413_large_right_right_right : List Nat := row12413_block_012 ++ row12413_block_013
theorem row12413_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12413 row12413_large_right_right_right :=
  Budget.RowUpperBound.branch row12413_block_012_covers row12413_block_013_covers

def row12413_large_right_right : List Nat := row12413_block_011 ++ row12413_large_right_right_right
theorem row12413_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12413 row12413_large_right_right :=
  Budget.RowUpperBound.branch row12413_block_011_covers row12413_large_right_right_right_covers

def row12413_large_right : List Nat := row12413_large_right_left ++ row12413_large_right_right
theorem row12413_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12413 row12413_large_right :=
  Budget.RowUpperBound.branch row12413_large_right_left_covers row12413_large_right_right_covers

def row12413_large : List Nat := row12413_large_left ++ row12413_large_right
theorem row12413_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12413 row12413_large :=
  Budget.RowUpperBound.branch row12413_large_left_covers row12413_large_right_covers

def row12413 : List Nat := row12413_large
theorem row12413_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12413 row12413 := row12413_large_covers

end ProjectEulerSolutions.PrimePairLarge
