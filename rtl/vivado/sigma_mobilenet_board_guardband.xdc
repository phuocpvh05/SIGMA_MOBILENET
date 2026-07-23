# SIGMA MobileNet board-release timing guardband.
#
# The PS still generates and the board still runs at 300 MHz (3.333 ns).  This
# setup uncertainty reserves 0.200 ns inside that period, so a timing PASS
# requires the routed data paths to have at least 200 ps of physical margin.
# MOB6 also reserves 0.050 ns for hold.  MOB5 formally passed with only 9 ps
# hold and 13 ps setup slack after guardband, yet produced an incorrect result
# on the physical board.  XDC files accept constraint commands, not general Tcl
# control flow, so keep these as direct commands.
# PROCESSING_ORDER=LATE is assigned by the project/update scripts, after the PS
# IP creates clk_pl_0.
set_clock_uncertainty -setup 0.200 [get_clocks clk_pl_0]
set_clock_uncertainty -hold 0.050 [get_clocks clk_pl_0]
