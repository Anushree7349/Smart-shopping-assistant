from pydantic import BaseModel

# ---------------- MODELS ---------------- #

class Product(BaseModel):
    name: str
    price: float
    rating: float
    budget: float
    discount: float


class Observation(BaseModel):
    current_product: Product


class Action(BaseModel):
    action_type: str


class Reward(BaseModel):
    value: float


# ---------------- ENV ---------------- #

class SmartShoppingEnv:
    def __init__(self):
        self.products = []
        self.current_index = 0
        self.done = False
        self.history = []

    def reset(self):
        self.products = self._generate_products()
        self.current_index = 0
        self.done = False
        self.history = []
        return self._get_observation()

    def state(self):
        return {
            "index": self.current_index,
            "history": self.history
        }

    def _get_observation(self):
        if self.current_index >= len(self.products):
            return None
        return Observation(current_product=self.products[self.current_index])

    def step(self, action: Action):
        if self.done:
            return self._get_observation(), Reward(value=0.0), True, {}

        product = self.products[self.current_index]

        reward = self._calculate_reward(product, action.action_type)

        # Save history
        self.history.append({
            "product": product.name,
            "action": action.action_type,
            "reward": reward
        })

        self.current_index += 1

        if self.current_index >= len(self.products):
            self.done = True

        return self._get_observation(), Reward(value=reward), self.done, {}

    # ---------------- FIXED REWARD FUNCTION ---------------- #

    def _calculate_reward(self, product, action):
        reward = 0.0

        # 🎯 BUY
        if action == "buy":
            if product.rating >= 4:
                reward += 2
            if product.price <= product.budget:
                reward += 1
            if product.discount >= 20:
                reward += 1

        # 🎯 COMPARE
        elif action == "compare":
            reward += 0.8

        # 🎯 WISHLIST
        elif action == "wishlist":
            reward += 0.5

        # 🎯 SKIP
        elif action == "skip":
            if product.rating < 3:
                reward += 0.2
            else:
                reward -= 0.3

        return round(reward, 2)

    # ---------------- FIXED DATA ---------------- #

    def _generate_products(self):
        return [
            Product(name="Laptop", price=50000, rating=4.5, budget=60000, discount=30),
            Product(name="Phone", price=20000, rating=4.2, budget=25000, discount=25),
            Product(name="Headphones", price=2000, rating=3.8, budget=3000, discount=40),
            Product(name="Shoes", price=3000, rating=4.0, budget=4000, discount=20),
            Product(name="Watch", price=5000, rating=2.5, budget=4000, discount=10),
        ]