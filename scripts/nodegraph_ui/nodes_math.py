# scripts/nodegraph_ui/nodes_math.py
# Math nodes for the node graph UI, separated for organization

from .classes import ProcessorNode, InputSocket, OutputSocket, SocketType

class AddNode(ProcessorNode):
    @property
    def is_soft_computation(self):
        return True
    def __init__(self):
        super().__init__()
        self.display_name = "Add"
        self.category = "Math Nodes"
        self.description = "Adds two numeric inputs together."
        self.tooltips_in = {
            "a": "First numeric input.",
            "b": "Second numeric input."
        }
        self.tooltips_out = {
            "result": "Result of adding the two inputs."
        }

        self.inputs["a"] = InputSocket(
            self, "a", SocketType.FLOAT
        )

        self.inputs["b"] = InputSocket(
            self, "b", SocketType.FLOAT
        )

        self.outputs["result"] = OutputSocket(
            self, "result", SocketType.FLOAT
        )
    
    def compute(self):
        a = self.inputs["a"].get()
        b = self.inputs["b"].get()
        if a is None or b is None:
            self.outputs["result"]._cache = None
            return

        try:
            result = float(a) + float(b)
        except Exception:
            result = None

        self.outputs["result"]._cache = result

class SubtractNode(ProcessorNode):
    @property
    def is_soft_computation(self):
        return True
    def __init__(self):
        super().__init__()
        self.display_name = "Subtract"
        self.category = "Math Nodes"
        self.description = "Subtracts the second numeric input from the first."
        self.tooltips_in = {
            "a": "First numeric input.",
            "b": "Second numeric input."
        }
        self.tooltips_out = {
            "result": "Result of subtracting the second input from the first."
        }

        self.inputs["a"] = InputSocket(
            self, "a", SocketType.FLOAT
        )

        self.inputs["b"] = InputSocket(
            self, "b", SocketType.FLOAT
        )

        self.outputs["result"] = OutputSocket(
            self, "result", SocketType.FLOAT
        )
    
    def compute(self):
        a = self.inputs["a"].get()
        b = self.inputs["b"].get()
        if a is None or b is None:
            self.outputs["result"]._cache = None
            return

        try:
            result = float(a) - float(b)
        except Exception:
            result = None

        self.outputs["result"]._cache = result

class MultiplyNode(ProcessorNode):
    @property
    def is_soft_computation(self):
        return True
    def __init__(self):
        super().__init__()
        self.display_name = "Multiply"
        self.category = "Math Nodes"
        self.description = "Multiplies two numeric inputs together."
        self.tooltips_in = {
            "a": "First numeric input.",
            "b": "Second numeric input."
        }
        self.tooltips_out = {
            "result": "Result of multiplying the two inputs."
        }

        self.inputs["a"] = InputSocket(
            self, "a", SocketType.FLOAT
        )

        self.inputs["b"] = InputSocket(
            self, "b", SocketType.FLOAT
        )

        self.outputs["result"] = OutputSocket(
            self, "result", SocketType.FLOAT
        )
    
    def compute(self):
        a = self.inputs["a"].get()
        b = self.inputs["b"].get()
        if a is None or b is None:
            self.outputs["result"]._cache = None
            return

        try:
            result = float(a) * float(b)
        except Exception:
            result = None

        self.outputs["result"]._cache = result

class DivideNode(ProcessorNode):
    @property
    def is_soft_computation(self):
        return True
    def __init__(self):
        super().__init__()
        self.display_name = "Divide"
        self.category = "Math Nodes"
        self.description = "Divides the first numeric input by the second."
        self.tooltips_in = {
            "a": "Numerator numeric input.",
            "b": "Denominator numeric input."
        }
        self.tooltips_out = {
            "result": "Result of dividing the first input by the second."
        }

        self.inputs["a"] = InputSocket(
            self, "a", SocketType.FLOAT
        )

        self.inputs["b"] = InputSocket(
            self, "b", SocketType.FLOAT
        )

        self.outputs["result"] = OutputSocket(
            self, "result", SocketType.FLOAT
        )
    
    def compute(self):
        a = self.inputs["a"].get()
        b = self.inputs["b"].get()
        if a is None or b is None:
            self.outputs["result"]._cache = None
            return

        try:
            result = float(a) / float(b)
        except Exception:
            result = None

        self.outputs["result"]._cache = result

class ModuloNode(ProcessorNode):
    def __init__(self):
        super().__init__()
        self.display_name = "Modulo"
        self.category = "Math Nodes"
        self.description = "Computes the modulo of the first numeric input by the second."
        self.tooltips_in = {
            "a": "Dividend numeric input.",
            "b": "Divisor numeric input."
        }
        self.tooltips_out = {
            "result": "Result of the modulo operation."
        }

        self.inputs["a"] = InputSocket(
            self, "a", SocketType.FLOAT
        )

        self.inputs["b"] = InputSocket(
            self, "b", SocketType.FLOAT
        )

        self.outputs["result"] = OutputSocket(
            self, "result", SocketType.FLOAT
        )
    
    def compute(self):
        a = self.inputs["a"].get()
        b = self.inputs["b"].get()
        if a is None or b is None:
            self.outputs["result"]._cache = None
            return

        try:
            result = float(a) % float(b)
        except Exception:
            result = None

        self.outputs["result"]._cache = result

class PowerNode(ProcessorNode):
    def __init__(self):
        super().__init__()
        self.display_name = "Power"
        self.category = "Math Nodes"
        self.description = "Raises the first numeric input to the power of the second."
        self.tooltips_in = {
            "base": "Base numeric input.",
            "exponent": "Exponent numeric input."
        }
        self.tooltips_out = {
            "result": "Result of raising the base to the exponent."
        }

        self.inputs["base"] = InputSocket(
            self, "base", SocketType.FLOAT
        )

        self.inputs["exponent"] = InputSocket(
            self, "exponent", SocketType.FLOAT
        )

        self.outputs["result"] = OutputSocket(
            self, "result", SocketType.FLOAT
        )
    
    def compute(self):
        base = self.inputs["base"].get()
        exponent = self.inputs["exponent"].get()
        if base is None or exponent is None:
            self.outputs["result"]._cache = None
            return

        try:
            result = float(base) ** float(exponent)
        except Exception:
            result = None

        self.outputs["result"]._cache = result

class SqrtNode(ProcessorNode):
    def __init__(self):
        super().__init__()
        self.display_name = "Square Root"
        self.category = "Math Nodes"
        self.description = "Computes the square root of the numeric input."
        self.tooltips_in = {
            "value": "Numeric input to compute the square root of."
        }
        self.tooltips_out = {
            "result": "Result of the square root operation."
        }

        self.inputs["value"] = InputSocket(
            self, "value", SocketType.FLOAT
        )

        self.outputs["result"] = OutputSocket(
            self, "result", SocketType.FLOAT
        )
    
    def compute(self):
        value = self.inputs["value"].get()
        if value is None:
            self.outputs["result"]._cache = None
            return

        try:
            result = float(value) ** 0.5
        except Exception:
            result = None

        self.outputs["result"]._cache = result

class LogNode(ProcessorNode):
    def __init__(self):
        super().__init__()
        self.display_name = "Logarithm"
        self.category = "Math Nodes"
        self.description = "Computes the logarithm of the numeric input with the specified base."
        self.tooltips_in = {
            "value": "Numeric input to compute the logarithm of.",
            "base": "Base of the logarithm."
        }
        self.tooltips_out = {
            "result": "Result of the logarithm operation."
        }

        self.inputs["value"] = InputSocket(
            self, "value", SocketType.FLOAT
        )

        self.inputs["base"] = InputSocket(
            self, "base", SocketType.FLOAT
        )

        self.outputs["result"] = OutputSocket(
            self, "result", SocketType.FLOAT
        )
    
    def compute(self):
        value = self.inputs["value"].get()
        base = self.inputs["base"].get()
        if value is None or base is None:
            self.outputs["result"]._cache = None
            return

        try:
            import math
            result = math.log(float(value), float(base))
        except Exception:
            result = None

        self.outputs["result"]._cache = result

class CustomEquationNode(ProcessorNode):
    def __init__(self):
        super().__init__()
        self.display_name = "Custom Equation"
        self.category = "Math Nodes"
        self.description = "Evaluates a custom mathematical equation using the provided variable inputs."
        self.tooltips_in = {
            "equation": "Mathematical equation to evaluate (e.g., 'a + b * c').",
            "variables": "Dictionary of variable names and their numeric values."
        }
        self.tooltips_out = {
            "result": "Result of evaluating the custom equation."
        }

        self.inputs["equation"] = InputSocket(
            self, "equation", SocketType.STRING
        )

        try:
            self.inputs["variables"] = InputSocket(
                self, "variables", SocketType.DICT
            )
        except Exception:
            print("CustomEquationNode: SocketType.DICT not available, using UNDEFINED instead.")
            self.inputs["variables"] = InputSocket(
                self, "variables", SocketType.UNDEFINED
            )

        self.outputs["result"] = OutputSocket(
            self, "result", SocketType.FLOAT
        )
    
    def compute(self):
        equation = self.inputs["equation"].get()
        variables = self.inputs["variables"].get()
        if equation is None or variables is None:
            self.outputs["result"]._cache = None
            return

        try:
            # Safely evaluate the equation using the provided variables
            result = eval(equation, {"__builtins__": None}, variables)
        except Exception:
            result = None

        self.outputs["result"]._cache = result